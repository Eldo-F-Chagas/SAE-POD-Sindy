"""
Autocontained SAE + SINDy workflow for cavity1000.

This file contains the full reduced-order pipeline for:
1. loading SAE latent coordinates
2. discovering sparse equations with SINDy
3. generating trajectory comparisons and heatmaps
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso


@dataclass
class ReducedDataset:
    name: str
    state_symbol: str
    states: np.ndarray
    times: np.ndarray
    source: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SindyConfig:
    polynomial_order: int = 2
    use_sine: bool = True
    alpha_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    max_plot_states: int = 6
    heatmap_times: tuple[float, ...] = ()
    heatmap_grid_size: int = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAE+SINDy workflow.")
    parser.add_argument("--poly-order", type=int, default=2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--max-plot-states", type=int, default=6)
    parser.add_argument("--heatmap-times", type=float, nargs="*", default=[])
    parser.add_argument("--heatmap-grid-size", type=int, default=64)
    parser.add_argument("--no-sine", dest="use_sine", action="store_false")
    parser.set_defaults(use_sine=True)
    return parser.parse_args()


def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - estimate) / (np.linalg.norm(reference) + 1e-12))


def make_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def sort_and_deduplicate(times: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="stable")
    times = np.asarray(times[order], dtype=np.float64)
    states = np.asarray(states[order], dtype=np.float64)
    unique_times, unique_idx = np.unique(times, return_index=True)
    if len(unique_times) != len(times):
        times = times[unique_idx]
        states = states[unique_idx]
    return times, states


def infer_dt(times: np.ndarray) -> float:
    deltas = np.diff(times)
    return 0.0 if deltas.size == 0 else float(np.median(deltas))


def load_sae_dataset(case_dir: Path) -> ReducedDataset:
    latent_path = Path(case_dir) / "SAE" / "resultados" / "SAE" / "latents.npz"
    if not latent_path.exists():
        raise FileNotFoundError(f"SAE latent results not found: {latent_path}")

    data = np.load(latent_path, allow_pickle=True)
    if {"z_train", "z_val", "t_train", "t_val"}.issubset(set(data.files)):
        states = np.vstack(
            [
                np.asarray(data["z_train"], dtype=np.float64),
                np.asarray(data["z_val"], dtype=np.float64),
            ]
        )
        times = np.concatenate(
            [
                np.asarray(data["t_train"], dtype=np.float64),
                np.asarray(data["t_val"], dtype=np.float64),
            ]
        )
    elif "latent_data" in data.files:
        states = np.asarray(data["latent_data"], dtype=np.float64)
        times = (
            np.asarray(data["times"], dtype=np.float64)
            if "times" in data.files
            else np.arange(states.shape[0], dtype=np.float64)
        )
    else:
        raise KeyError("Unsupported SAE latent file format.")

    times, states = sort_and_deduplicate(times, states)
    return ReducedDataset(
        name="sae",
        state_symbol="z",
        states=states,
        times=times,
        source=latent_path,
        metadata={"latent_dim": int(states.shape[1]), "dt": infer_dt(times)},
    )


def generate_powers(n_variables: int, order: int) -> list[tuple[int, ...]]:
    if n_variables == 1:
        return [(power,) for power in range(order + 1)]
    powers: list[tuple[int, ...]] = []
    for power in range(order + 1):
        for suffix in generate_powers(n_variables - 1, order - power):
            powers.append((power,) + suffix)
    return sorted(powers, reverse=False)


@dataclass
class PolynomialLibrary:
    n_variables: int
    order: int
    use_sine: bool
    variable_symbol: str

    def __post_init__(self) -> None:
        self.powers = generate_powers(self.n_variables, self.order)
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> list[str]:
        names = []
        for power in self.powers:
            if sum(power) == 0:
                names.append("1")
                continue
            parts = []
            for index, exponent in enumerate(power):
                if exponent == 0:
                    continue
                variable = f"{self.variable_symbol}{index + 1}"
                parts.append(variable if exponent == 1 else f"{variable}^{exponent}")
            names.append("*".join(parts))
        if self.use_sine:
            for index in range(self.n_variables):
                variable = f"{self.variable_symbol}{index + 1}"
                names.append(f"sin({variable})")
                names.append(f"cos({variable})")
        return names

    def transform(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        theta = np.empty((states.shape[0], len(self.feature_names)), dtype=np.float64)
        for column, power in enumerate(self.powers):
            theta[:, column] = np.prod(np.power(states, power), axis=1)
        next_column = len(self.powers)
        if self.use_sine:
            for index in range(self.n_variables):
                theta[:, next_column] = np.sin(states[:, index])
                theta[:, next_column + 1] = np.cos(states[:, index])
                next_column += 2
        return theta

    def transform_single(self, state: np.ndarray) -> np.ndarray:
        return self.transform(np.asarray(state, dtype=np.float64)[None, :])[0]


def compute_derivatives(states: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.gradient(states, times, axis=0, edge_order=2 if len(times) > 2 else 1)


def fit_sparse_model(theta: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    scale = np.linalg.norm(theta, axis=0)
    scale[scale < 1e-12] = 1.0
    theta_scaled = theta / scale[None, :]
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000, tol=1e-5)
    model.fit(theta_scaled, target)
    return model.coef_ / scale


def discover_equations(theta: np.ndarray, derivatives: np.ndarray, alpha_values: tuple[float, ...]):
    print(f"   Library size: {theta.shape[1]} features")
    print(f"   Testing {len(alpha_values)} alpha values")

    n_states = derivatives.shape[1]
    results = []
    for alpha in alpha_values:
        coefficients = np.zeros((theta.shape[1], n_states), dtype=np.float64)
        for index in range(n_states):
            coefficients[:, index] = fit_sparse_model(theta, derivatives[:, index], alpha)
        derivatives_pred = theta @ coefficients
        result = {
            "alpha": float(alpha),
            "Xi": coefficients,
            "derivatives_pred": derivatives_pred,
            "derivative_l2_error": relative_l2(derivatives, derivatives_pred),
            "n_nonzero": int(np.count_nonzero(np.abs(coefficients) > 1e-8)),
        }
        print(
            f"   alpha={alpha:.4g} | derivative L2={result['derivative_l2_error']:.6f} "
            f"| nonzero terms={result['n_nonzero']}"
        )
        results.append(result)

    best_result = min(results, key=lambda item: item["derivative_l2_error"])
    print(
        f"   Selected alpha={best_result['alpha']:.4g} "
        f"with derivative L2={best_result['derivative_l2_error']:.6f}"
    )
    return best_result, results


def simulate_sindy(
    library: PolynomialLibrary,
    coefficients: np.ndarray,
    times: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        return library.transform_single(state) @ coefficients

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        np.asarray(initial_state, dtype=np.float64),
        t_eval=np.asarray(times, dtype=np.float64),
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    if not solution.success or solution.y.shape[1] != len(times):
        return None, str(solution.message)
    return solution.y.T, None


def simulate_reference_guided_sindy(
    library: PolynomialLibrary,
    coefficients: np.ndarray,
    times: np.ndarray,
    reference_states: np.ndarray,
) -> np.ndarray:
    predicted = np.zeros_like(reference_states, dtype=np.float64)
    predicted[0] = reference_states[0]
    for index in range(1, len(times)):
        dt = float(times[index] - times[index - 1])
        rhs = library.transform_single(reference_states[index - 1]) @ coefficients
        predicted[index] = reference_states[index - 1] + dt * rhs
    return predicted


def format_equations(coefficients: np.ndarray, feature_names: list[str], state_symbol: str) -> str:
    lines = []
    for state_index in range(coefficients.shape[1]):
        parts = []
        for value, name in zip(coefficients[:, state_index], feature_names):
            if abs(value) <= 1e-8:
                continue
            parts.append(f"{value:+.6e}" if name == "1" else f"{value:+.6e}*{name}")
        lines.append(f"d{state_symbol}{state_index + 1}/dt = {' '.join(parts) if parts else '0.0'}")
    return "\n".join(lines)


def plot_alpha_scan(scan_results: list[dict[str, Any]], output_dir: Path, title: str) -> None:
    alphas = np.array([item["alpha"] for item in scan_results], dtype=np.float64)
    deriv_errors = np.array([item["derivative_l2_error"] for item in scan_results], dtype=np.float64)
    nonzero = np.array([item["n_nonzero"] for item in scan_results], dtype=np.float64)

    fig, axis_left = plt.subplots(figsize=(8, 4.5))
    axis_left.plot(alphas, deriv_errors, marker="o", linewidth=2, color="tab:blue")
    axis_left.set_xscale("log")
    axis_left.set_xlabel("alpha")
    axis_left.set_ylabel("Derivative relative L2", color="tab:blue")
    axis_left.tick_params(axis="y", labelcolor="tab:blue")
    axis_left.grid(True, alpha=0.3)

    axis_right = axis_left.twinx()
    axis_right.plot(alphas, nonzero, marker="s", linewidth=2, color="tab:red")
    axis_right.set_ylabel("Nonzero terms", color="tab:red")
    axis_right.tick_params(axis="y", labelcolor="tab:red")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_scan.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_state_comparison(
    times: np.ndarray,
    states: np.ndarray,
    states_pred: np.ndarray | None,
    label_reference: str,
    label_model: str,
    output_dir: Path,
    state_symbol: str,
    max_states: int,
) -> None:
    n_states = min(max_states, states.shape[1])
    fig, axes = plt.subplots(n_states, 1, figsize=(12, 3 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]
    for index in range(n_states):
        axes[index].plot(times, states[:, index], linewidth=2, color="tab:blue", label=label_reference)
        if states_pred is not None:
            axes[index].plot(
                times,
                states_pred[:, index],
                linestyle="--",
                linewidth=2,
                color="tab:orange",
                label=label_model,
            )
        axes[index].set_ylabel(f"{state_symbol}{index + 1}")
        axes[index].grid(True, alpha=0.3)
        axes[index].legend(loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"{label_model} trajectory vs {label_reference}")
    fig.tight_layout()
    plt.savefig(output_dir / "trajectory_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_derivative_comparison(
    times: np.ndarray,
    derivatives: np.ndarray,
    derivatives_pred: np.ndarray,
    output_dir: Path,
    state_symbol: str,
    max_states: int,
) -> None:
    n_states = min(max_states, derivatives.shape[1])
    fig, axes = plt.subplots(n_states, 1, figsize=(12, 3 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]
    for index in range(n_states):
        axes[index].plot(times, derivatives[:, index], linewidth=2, color="tab:green", label="Reference")
        axes[index].plot(
            times,
            derivatives_pred[:, index],
            linestyle="--",
            linewidth=2,
            color="tab:red",
            label="SINDy",
        )
        axes[index].set_ylabel(f"d{state_symbol}{index + 1}/dt")
        axes[index].grid(True, alpha=0.3)
        axes[index].legend(loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Derivative comparison")
    fig.tight_layout()
    plt.savefig(output_dir / "derivative_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_state_heatmap(
    times: np.ndarray,
    states: np.ndarray,
    output_dir: Path,
    state_symbol: str,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 4.5))
    image = axis.imshow(
        states.T,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 1, states.shape[1]],
        cmap="viridis",
    )
    axis.set_xlabel("Time")
    axis.set_ylabel(f"{state_symbol} index")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Amplitude")
    plt.tight_layout()
    plt.savefig(output_dir / "reduced_state_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def match_requested_times(
    times: np.ndarray,
    target_times: tuple[float, ...],
) -> tuple[list[dict[str, float]], list[float], float]:
    dt = infer_dt(times)
    tolerance = max(1e-8, 0.51 * abs(dt) if dt > 0 else 1e-8)
    matched: list[dict[str, float]] = []
    missing: list[float] = []
    for target in target_times:
        index = int(np.argmin(np.abs(times - target)))
        actual = float(times[index])
        if abs(actual - target) <= tolerance:
            matched.append({"requested_time": float(target), "actual_time": actual, "index": index})
        else:
            missing.append(float(target))
    return matched, missing, tolerance


def infer_square_grid_from_vector(vector_size: int, channels: int = 2) -> tuple[int, int]:
    if vector_size % channels != 0:
        raise ValueError(f"Vector size {vector_size} is not divisible by channels={channels}")
    pixels_per_channel = vector_size // channels
    side = int(np.sqrt(pixels_per_channel))
    if side * side != pixels_per_channel:
        raise ValueError(f"Cannot infer square grid from vector size {vector_size}")
    return side, side


def build_sae_snapshot_reconstructor(dataset: ReducedDataset):
    import tensorflow as tf

    results_dir = dataset.source.parent
    model_path = results_dir / "sae_dense_best.keras"
    if not model_path.exists():
        model_path = results_dir / "sae_dense.keras"
    if not model_path.exists():
        raise FileNotFoundError("Could not find a saved SAE model for reconstruction.")

    norm_path = results_dir / "sae_dense_norm.npz"
    if not norm_path.exists():
        raise FileNotFoundError("Could not find sae_dense_norm.npz for denormalization.")

    autoencoder = tf.keras.models.load_model(model_path, compile=False)
    norm_data = np.load(norm_path, allow_pickle=True)
    mu = np.asarray(norm_data["mu"], dtype=np.float64).reshape(-1)
    sig = np.asarray(norm_data["sig"], dtype=np.float64).reshape(-1)
    ny, nx = infer_square_grid_from_vector(mu.size, channels=2)

    latent_input = tf.keras.Input(shape=(dataset.states.shape[1],), name="latent_input")
    x = latent_input
    for layer_name in ("dec_dense_0", "dec_bn_0", "dec_dense_1", "dec_bn_1", "x_hat"):
        x = autoencoder.get_layer(layer_name)(x)
    decoder = tf.keras.Model(latent_input, x, name="sae_decoder")

    def reconstruct_batch(reduced_states: np.ndarray) -> np.ndarray:
        reduced_states = np.atleast_2d(np.asarray(reduced_states, dtype=np.float32))
        decoded_norm = decoder.predict(reduced_states, verbose=0)
        decoded = decoded_norm * sig[None, :] + mu[None, :]
        return decoded.reshape((-1, ny, nx, 2)).astype(np.float64)

    return reconstruct_batch


def save_velocity_snapshot(frame: np.ndarray, output_path: Path, title: str) -> None:
    ux = frame[..., 0]
    uy = frame[..., 1]
    umag = np.sqrt(ux**2 + uy**2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, values, label in zip(axes, (ux, uy, umag), ("Ux", "Uy", "|U|")):
        image = axis.imshow(values, origin="lower", cmap="viridis")
        axis.set_title(label)
        axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_velocity_comparison(
    reference_frame: np.ndarray,
    predicted_frame: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    ref_ux = reference_frame[..., 0]
    ref_uy = reference_frame[..., 1]
    ref_umag = np.sqrt(ref_ux**2 + ref_uy**2)
    pred_ux = predicted_frame[..., 0]
    pred_uy = predicted_frame[..., 1]
    pred_umag = np.sqrt(pred_ux**2 + pred_uy**2)

    panels = (
        (ref_ux, pred_ux, pred_ux - ref_ux, "Ux"),
        (ref_uy, pred_uy, pred_uy - ref_uy, "Uy"),
        (ref_umag, pred_umag, pred_umag - ref_umag, "|U|"),
    )

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row, (ref_values, pred_values, err_values, label) in enumerate(panels):
        for col, values, suffix in (
            (0, ref_values, "ref"),
            (1, pred_values, "SINDy"),
            (2, err_values, "err"),
        ):
            image = axes[row, col].imshow(values, origin="lower", cmap="viridis")
            axes[row, col].set_title(f"{label} {suffix}")
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_requested_time_heatmaps(
    dataset: ReducedDataset,
    trajectory_pred: np.ndarray | None,
    trajectory_method: str | None,
    output_dir: Path,
    config: SindyConfig,
) -> dict[str, Any] | None:
    if not config.heatmap_times:
        return None

    print(f"   Generating heatmaps for requested times: {config.heatmap_times}")
    matched, missing, tolerance = match_requested_times(dataset.times, config.heatmap_times)
    heatmap_dir = output_dir / "heatmaps_exact_times"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    reconstruct_batch = build_sae_snapshot_reconstructor(dataset)
    if matched:
        indices = [int(item["index"]) for item in matched]
        reference_frames = reconstruct_batch(dataset.states[indices])
        predicted_frames = reconstruct_batch(trajectory_pred[indices]) if trajectory_pred is not None else None

        for position, item in enumerate(matched):
            requested = float(item["requested_time"])
            actual = float(item["actual_time"])
            target_file = heatmap_dir / f"time_{requested:04.1f}s.png"
            if predicted_frames is None:
                title = f"SAE reconstruction | requested={requested:.3f}s | used={actual:.3f}s"
                save_velocity_snapshot(reference_frames[position], target_file, title)
            else:
                field_error = relative_l2(reference_frames[position], predicted_frames[position])
                sindy_label = "SINDy" if trajectory_method is None else f"SINDy ({trajectory_method})"
                title = (
                    f"SAE vs {sindy_label} | requested={requested:.3f}s | "
                    f"used={actual:.3f}s | relL2={field_error:.4f}"
                )
                save_velocity_comparison(
                    reference_frames[position],
                    predicted_frames[position],
                    target_file,
                    title,
                )

    report = {
        "requested_times": [float(value) for value in config.heatmap_times],
        "matched_times": matched,
        "missing_times": [float(value) for value in missing],
        "time_tolerance": float(tolerance),
        "comparison_with_sindy": bool(trajectory_pred is not None),
        "trajectory_method": trajectory_method,
    }
    (heatmap_dir / "report.json").write_text(json.dumps(make_json_ready(report), indent=2), encoding="utf-8")
    if missing:
        print(f"   Missing exact times for SAE: {missing}")
    return report


def save_results(
    dataset: ReducedDataset,
    library: PolynomialLibrary,
    best_result: dict[str, Any],
    scan_results: list[dict[str, Any]],
    trajectory_pred: np.ndarray | None,
    trajectory_error: float | None,
    integration_message: str | None,
    trajectory_method: str | None,
    derivative_best_alpha: float,
    output_dir: Path,
    config: SindyConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    equations = format_equations(best_result["Xi"], library.feature_names, dataset.state_symbol)
    (output_dir / "equations.txt").write_text(equations, encoding="utf-8")

    plot_state_heatmap(dataset.times, dataset.states, output_dir, dataset.state_symbol, "SAE reduced coordinates")
    plot_alpha_scan(scan_results, output_dir, title="SAE + SINDy alpha scan")
    plot_derivative_comparison(
        dataset.times,
        best_result["derivatives_reference"],
        best_result["derivatives_pred"],
        output_dir,
        dataset.state_symbol,
        config.max_plot_states,
    )
    plot_state_comparison(
        dataset.times,
        dataset.states,
        trajectory_pred,
        "Reference",
        "SINDy" if trajectory_method is None else f"SINDy ({trajectory_method})",
        output_dir,
        dataset.state_symbol,
        config.max_plot_states,
    )
    heatmap_report = generate_requested_time_heatmaps(
        dataset,
        trajectory_pred,
        trajectory_method,
        output_dir,
        config,
    )

    summary = {
        "architecture": "SAE+SINDy",
        "source_file": str(dataset.source),
        "n_snapshots": int(dataset.states.shape[0]),
        "n_coordinates": int(dataset.states.shape[1]),
        "dt": dataset.metadata.get("dt"),
        "polynomial_order": int(config.polynomial_order),
        "use_sine": bool(config.use_sine),
        "selected_alpha": float(best_result["alpha"]),
        "derivative_best_alpha": float(derivative_best_alpha),
        "library_size": int(best_result["Xi"].shape[0]),
        "derivative_l2_error": float(best_result["derivative_l2_error"]),
        "trajectory_l2_error": None if trajectory_error is None else float(trajectory_error),
        "trajectory_method": trajectory_method,
        "n_nonzero_terms": int(best_result["n_nonzero"]),
        "integration_message": integration_message,
        "metadata": make_json_ready(dataset.metadata),
        "heatmap_examples": make_json_ready(heatmap_report),
    }

    np.savez_compressed(
        output_dir / "sindy_results.npz",
        times=dataset.times,
        states=dataset.states,
        Xi=best_result["Xi"],
        feature_names=np.asarray(library.feature_names, dtype=object),
        alpha=np.array([best_result["alpha"]], dtype=np.float64),
        derivative_l2_error=np.array([best_result["derivative_l2_error"]], dtype=np.float64),
        n_nonzero=np.array([best_result["n_nonzero"]], dtype=np.int64),
        derivatives_reference=best_result["derivatives_reference"],
        derivatives_pred=best_result["derivatives_pred"],
        trajectory_pred=np.full_like(dataset.states, np.nan) if trajectory_pred is None else trajectory_pred,
    )
    (output_dir / "summary.json").write_text(json.dumps(make_json_ready(summary), indent=2), encoding="utf-8")
    return summary


def run_sae_analysis(
    case_dir: Path,
    config: SindyConfig | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    case_dir = Path(case_dir).resolve()
    output_root = case_dir / "SINDy" / "resultados" if output_root is None else Path(output_root)
    output_dir = output_root / "sae_sindy"
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = SindyConfig()

    dataset = load_sae_dataset(case_dir)

    print("\nProcessing SAE + SINDy")
    print(f"   Source: {dataset.source}")
    print(f"   Reduced coordinates: {dataset.states.shape}")
    print(f"   Time range: [{dataset.times[0]:.6f}, {dataset.times[-1]:.6f}]")

    derivatives = compute_derivatives(dataset.states, dataset.times)
    library = PolynomialLibrary(
        n_variables=dataset.states.shape[1],
        order=config.polynomial_order,
        use_sine=config.use_sine,
        variable_symbol=dataset.state_symbol,
    )
    theta = library.transform(dataset.states)
    best_result, scan_results = discover_equations(theta, derivatives, config.alpha_values)
    derivative_best_alpha = float(best_result["alpha"])

    for result in [best_result, *scan_results]:
        result["derivatives_reference"] = derivatives

    selected_result = best_result
    trajectory_method = "solve_ivp"
    trajectory_pred, integration_message = simulate_sindy(
        library,
        selected_result["Xi"],
        dataset.times,
        dataset.states[0],
    )

    if trajectory_pred is None:
        print(f"   Integration warning: {integration_message}")
        for result in scan_results:
            result["reference_guided_pred"] = simulate_reference_guided_sindy(
                library,
                result["Xi"],
                dataset.times,
                dataset.states,
            )
            result["reference_guided_l2_error"] = relative_l2(
                dataset.states,
                result["reference_guided_pred"],
            )
        selected_result = min(scan_results, key=lambda item: item["reference_guided_l2_error"])
        selected_result["derivatives_reference"] = derivatives
        trajectory_pred = selected_result["reference_guided_pred"]
        trajectory_error = float(selected_result["reference_guided_l2_error"])
        trajectory_method = "reference_guided_one_step"
        integration_message = f"{integration_message} | Fallback: reference-guided one-step SINDy"
        print(
            f"   Fallback alpha={selected_result['alpha']:.4g} "
            f"with trajectory relative L2={trajectory_error:.6f}"
        )
    else:
        trajectory_error = relative_l2(dataset.states, trajectory_pred)
        print(f"   Trajectory relative L2: {trajectory_error:.6f}")

    summary = save_results(
        dataset,
        library,
        selected_result,
        scan_results,
        trajectory_pred,
        trajectory_error,
        integration_message,
        trajectory_method,
        derivative_best_alpha,
        output_dir,
        config,
    )
    print(f"   Results saved to: {output_dir}")
    return summary


def main() -> None:
    args = parse_args()
    src_dir = Path(__file__).resolve().parent
    case_dir = src_dir.parent.parent
    config = SindyConfig(
        polynomial_order=int(args.poly_order),
        use_sine=bool(args.use_sine),
        alpha_values=tuple(float(value) for value in args.alphas),
        max_plot_states=int(args.max_plot_states),
        heatmap_times=tuple(float(value) for value in args.heatmap_times),
        heatmap_grid_size=int(args.heatmap_grid_size),
    )
    run_sae_analysis(case_dir, config)


if __name__ == "__main__":
    main()
