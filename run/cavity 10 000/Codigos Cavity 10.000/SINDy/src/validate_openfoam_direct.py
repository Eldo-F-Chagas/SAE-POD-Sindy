"""Validação temporal direta OpenFOAM versus SAE--POD--SINDy para cavity.

Não usa snapshots POD como referência, não supõe tempo normalizado em [0, 1]
e não procura campos de dam-break. Os estados OpenFOAM são lidos diretamente
nos diretórios temporais mais próximos dos tempos da trajetória SINDy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from fluidfoam import readfield, readmesh
from scipy.interpolate import griddata

CASE_DIR = Path(r"/home/mlep-pc-ubuntu/OpenFOAM/mlep-pc-ubuntu-13/run/cavity 10 000")
CODE_DIR = CASE_DIR / "Codigos Cavity 10.000"
RUN_DIR = CODE_DIR / "SINDy/resultados/sae_pod_sindy/order1_time_alasso"
OUT_DIR = RUN_DIR / "openfoam_direct_vs_sindy"
GRID_NX = GRID_NY = 48
FIELDS = ("|U|", "p", "T", "k")


def numeric_time_dirs(case_dir: Path) -> tuple[list[str], np.ndarray]:
    names = []
    for path in case_dir.iterdir():
        if path.is_dir():
            try:
                float(path.name)
            except ValueError:
                continue
            names.append(path.name)
    names.sort(key=float)
    if not names:
        raise RuntimeError(f"Nenhum diretório temporal em {case_dir}")
    return names, np.asarray([float(name) for name in names], dtype=float)


def inverse_pod_to_latents(q_scaled: np.ndarray, model: Any) -> np.ndarray:
    q_scaled = np.atleast_2d(np.asarray(q_scaled, dtype=np.float64))
    q = q_scaled * model["q_std"][None, :] + model["q_mean"][None, :]
    z_norm = q @ model["components"] + model["pod_mean"][None, :]
    return z_norm * model["z_std"][None, :] + model["z_mean"][None, :]


def build_decoder():
    import tensorflow as tf

    results = CODE_DIR / "SAE/resultados"
    model_path = results / "sae_dense_best.keras"
    if not model_path.exists():
        model_path = results / "sae_dense.keras"
    autoencoder = tf.keras.models.load_model(model_path, compile=False)
    norm = np.load(results / "sae_dense_norm.npz", allow_pickle=True)
    mu = np.asarray(norm["mu"], dtype=np.float64).reshape(-1)
    sig = np.asarray(norm["sig"], dtype=np.float64).reshape(-1)
    names = [str(v) for v in np.asarray(norm["channel_names"]).ravel()]
    if names != ["Ux", "Uy", "p", "T", "k"]:
        raise ValueError(f"Canais SAE inesperados: {names}")

    layers = [layer.name for layer in autoencoder.layers]
    latent = autoencoder.get_layer("latent")
    x_in = tf.keras.Input(shape=(latent.output.shape[-1],), name="latent_validation_input")
    x = x_in
    for name in layers[layers.index("latent") + 1:]:
        x = autoencoder.get_layer(name)(x)
    decoder = tf.keras.Model(x_in, x)

    def decode(z: np.ndarray) -> np.ndarray:
        pred_n = decoder.predict(np.atleast_2d(z).astype(np.float32), verbose=0)
        pred = pred_n * sig[None, :] + mu[None, :]
        return pred.reshape((-1, GRID_NY, GRID_NX, len(names)))

    return decode, names


def vector_components(value: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value, dtype=float)
    if value.shape == (n_cells, 3):
        return value[:, 0], value[:, 1]
    if value.shape == (3, n_cells):
        return value[0], value[1]
    raise ValueError(f"Formato inesperado para U: {value.shape}")


def scalar_values(value: np.ndarray, n_cells: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float).squeeze()
    if value.ndim == 0:
        return np.full(n_cells, float(value))
    if value.size == n_cells:
        return value.reshape(-1)
    raise ValueError(f"Formato inesperado para {name}: {value.shape}")


def interpolate(values: np.ndarray, x: np.ndarray, y: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    points = np.column_stack((x, y))
    out = griddata(points, values, (gx, gy), method="linear")
    if np.isnan(out).any():
        nearest = griddata(points, values, (gx, gy), method="nearest")
        out = np.where(np.isnan(out), nearest, out)
    return np.asarray(out, dtype=float)


def read_openfoam_frame(time_name: str, x: np.ndarray, y: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> dict[str, np.ndarray]:
    n = x.size
    ux, uy = vector_components(readfield(str(CASE_DIR), time_name, "U"), n)
    result = {"|U|": interpolate(np.sqrt(ux**2 + uy**2), x, y, gx, gy)}
    for name in ("p", "T", "k"):
        values = scalar_values(readfield(str(CASE_DIR), time_name, name), n, name)
        result[name] = interpolate(values, x, y, gx, gy)
    return result


def decoded_fields(frame: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    index = {name: i for i, name in enumerate(names)}
    ux, uy = frame[..., index["Ux"]], frame[..., index["Uy"]]
    return {
        "|U|": np.sqrt(ux**2 + uy**2),
        "p": frame[..., index["p"]],
        "T": frame[..., index["T"]],
        "k": frame[..., index["k"]],
    }


def rel_l2(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))


def centered_rel_l2(ref: np.ndarray, pred: np.ndarray) -> float:
    ref_c, pred_c = ref - np.mean(ref), pred - np.mean(pred)
    return float(np.linalg.norm(pred_c - ref_c) / (np.linalg.norm(ref_c) + 1e-12))


def r2(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((pred - ref) ** 2) / (np.sum((ref - np.mean(ref)) ** 2) + 1e-12))


def plot_frame(reference: dict[str, np.ndarray], prediction: dict[str, np.ndarray], path: Path, title: str) -> dict[str, float]:
    fig, axes = plt.subplots(len(FIELDS), 3, figsize=(12, 3.2 * len(FIELDS)))
    metrics: dict[str, float] = {}
    for row, name in enumerate(FIELDS):
        ref, pred = reference[name], prediction[name]
        err = pred - ref
        lo, hi = float(min(ref.min(), pred.min())), float(max(ref.max(), pred.max()))
        lim = max(float(np.max(np.abs(err))), 1e-12)
        values = {"rel_l2": rel_l2(ref, pred), "centered_rel_l2": centered_rel_l2(ref, pred), "r2": r2(ref, pred)}
        for metric, value in values.items():
            metrics[f"{name}_{metric}"] = value
        panels = ((ref, f"{name} OpenFOAM", "viridis", lo, hi),
                  (pred, f"{name} SINDy/SAE", "viridis", lo, hi),
                  (err, f"erro | L2={values['rel_l2']:.3g} | R²={values['r2']:.3g}", "coolwarm", -lim, lim))
        for col, (image, label, cmap, vmin, vmax) in enumerate(panels):
            im = axes[row, col].imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            axes[row, col].set_title(label, fontsize=9)
            axes[row, col].axis("off")
            fig.colorbar(im, ax=axes[row, col], fraction=.046, pad=.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return metrics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = np.load(RUN_DIR / "best_model.npz", allow_pickle=True)
    times_sindy = np.asarray(model["times"], dtype=float)
    if "free_pred" not in model.files or not np.isfinite(model["free_pred"]).all():
        raise RuntimeError("O modelo não contém uma trajetória livre finita.")
    z_prediction = inverse_pod_to_latents(model["free_pred"], model)
    decode, channel_names = build_decoder()

    time_names, times_openfoam = numeric_time_dirs(CASE_DIR)
    xc, yc, _ = readmesh(str(CASE_DIR))
    x, y = np.asarray(xc, dtype=float).ravel(), np.asarray(yc, dtype=float).ravel()
    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), GRID_NX), np.linspace(y.min(), y.max(), GRID_NY))

    targets = np.linspace(times_sindy.min(), times_sindy.max(), 5)
    report = []
    for target in targets:
        sindy_index = int(np.argmin(np.abs(times_sindy - target)))
        openfoam_index = int(np.argmin(np.abs(times_openfoam - times_sindy[sindy_index])))
        sindy_time = float(times_sindy[sindy_index])
        openfoam_time = float(times_openfoam[openfoam_index])
        reference = read_openfoam_frame(time_names[openfoam_index], x, y, grid_x, grid_y)
        prediction = decoded_fields(decode(z_prediction[sindy_index:sindy_index + 1])[0], channel_names)
        output_file = OUT_DIR / f"time_{sindy_time:.6f}s.png"
        metrics = plot_frame(reference, prediction, output_file, f"SINDy t={sindy_time:.6f} s | OpenFOAM t={openfoam_time:.6f} s")
        report.append({
            "target_time": float(target), "sindy_index": sindy_index, "sindy_time": sindy_time,
            "openfoam_index": openfoam_index, "openfoam_time": openfoam_time,
            "time_mismatch": abs(openfoam_time - sindy_time), "file": str(output_file), **metrics,
        })
        print(report[-1])

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "README.txt").write_text(
        "Referência lida diretamente dos campos U, p, T e k do OpenFOAM.\n"
        "Alinhamento pelo tempo físico real mais próximo; sem dados POD e sem hipótese [0,1].\n",
        encoding="utf-8",
    )
    print(f"Relatório: {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
