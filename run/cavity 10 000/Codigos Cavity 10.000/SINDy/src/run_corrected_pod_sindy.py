# Esta célula importa bibliotecas numéricas, visualização, regressão esparsa e integração temporal usadas no notebook.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
# Força backend não-interativo para evitar conflito matplotlib/matplotlib_inline no Jupyter.
# Este erro aparece como: AttributeError: 'RcParams' object has no attribute '_get'.
matplotlib.use("Agg", force=True)
matplotlib.rcParams["backend"] = "Agg"

import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Função auxiliar de robustez: força backend Agg antes de criar figuras.
# Evita conflito entre matplotlib antigo do sistema e matplotlib_inline novo do Jupyter.
def force_matplotlib_agg() -> None:
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["backend"] = "Agg"
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass


# Esta célula define os caminhos do diretório atual, do caso OpenFOAM e da pasta de resultados SINDy.
SRC_DIR = Path(r"/home/mlep-pc-ubuntu/OpenFOAM/mlep-pc-ubuntu-13/run/cavity 10 000/Codigos Cavity 10.000/SINDy/src").resolve()
CASE_DIR = SRC_DIR.parent.parent
# Resultados corrigidos ficam separados dos artefatos legados.
OUTPUT_ROOT = CASE_DIR / "SINDy" / "resultados_corrigidos"

print(f"SRC_DIR     = {SRC_DIR}")
print(f"CASE_DIR    = {CASE_DIR}")
print(f"OUTPUT_ROOT = {OUTPUT_ROOT}")


# Esta célula define estruturas de dados e funções para ler e preparar as coordenadas reduzidas.
# Classe de dados: guarda nome do caso reduzido, símbolo das variáveis, estados reduzidos, tempos, arquivo-fonte e metadados.
@dataclass
class ReducedDataset:
    name: str
    state_symbol: str
    states: np.ndarray
    times: np.ndarray
    source: Path
    metadata: dict[str, Any] = field(default_factory=dict)


# Classe de configuração: concentra os parâmetros do ajuste SINDy e das figuras para facilitar mudanças em uma única célula.
@dataclass
class SindyConfig:
    polynomial_order: int = 2
    use_sine: bool = True
    alpha_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    pod_modes: int = 4
    max_plot_states: int = 6
    heatmap_times: tuple[float, ...] = ()
    heatmap_grid_size: int = 64



# Função de erro: calcula o erro relativo L2 entre a referência e a estimativa.
def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - estimate) / (np.linalg.norm(reference) + 1e-12))


# Função auxiliar: converte objetos NumPy/listas/dicionários para tipos serializáveis em JSON.
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


# Função auxiliar POD: converte tempos lidos do arquivo para floats.
def to_float_times(raw_times: np.ndarray) -> np.ndarray:
    return np.asarray([float(str(value).strip()) for value in np.asarray(raw_times).ravel()], dtype=np.float64)


# Função auxiliar POD: garante que a matriz de coeficientes tenha formato (n_tempos, n_modos).
def ensure_time_major(data: np.ndarray, n_times: int) -> np.ndarray:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[0] == n_times:
        return array
    if array.shape[1] == n_times:
        return array.T
    raise ValueError(f"Could not align reduced coordinates with n_times={n_times}; shape={array.shape}")


# Função auxiliar: ordena os snapshots pelo tempo e remove tempos duplicados mantendo a primeira ocorrência.
def sort_and_deduplicate(times: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="stable")
    times = np.asarray(times[order], dtype=np.float64)
    states = np.asarray(states[order], dtype=np.float64)
    unique_times, unique_idx = np.unique(times, return_index=True)
    if len(unique_times) != len(times):
        times = times[unique_idx]
        states = states[unique_idx]
    return times, states


# Função auxiliar: estima o passo temporal típico usando a mediana de diff(times).
def infer_dt(times: np.ndarray) -> float:
    deltas = np.diff(times)
    return 0.0 if deltas.size == 0 else float(np.median(deltas))


# Função de carregamento POD: lê pod_results.npz, extrai os coeficientes temporais e seleciona os modos desejados.
def load_pod_dataset(case_dir: Path, pod_modes: int) -> ReducedDataset:
    # Procura o arquivo POD em nomes usados por diferentes versões do processamento.
    # Primeiro tenta o nome antigo pod_results.npz; depois os nomes reais encontrados nesta pasta.
    pod_candidates = (
        Path(case_dir) / "POD" / "resultados_corrigidos" / "pod_results.npz",
        Path(case_dir) / "POD" / "resultados" / "pod_results.npz",
        Path(case_dir) / "POD" / "resultados" / "resultados_pod_completos.npz",
        Path(case_dir) / "POD" / "resultados" / "resultados_pod_basicos.npz",
    )
    pod_path = next((path for path in pod_candidates if path.exists()), None)
    if pod_path is None:
        checked = "\n".join(str(path) for path in pod_candidates)
        raise FileNotFoundError(f"POD results not found. Checked:\n{checked}")

    data = np.load(pod_path, allow_pickle=True)
    times = to_float_times(data["times"])

    if "A" in data.files:
        coefficients = ensure_time_major(data["A"], len(times))
        singular_values = np.asarray(data["s"], dtype=np.float64) if "s" in data.files else None
    else:
        singular_values = np.asarray(data["s"], dtype=np.float64)
        coefficients = (singular_values[:, None] * np.asarray(data["Vt"], dtype=np.float64)).T

    n_modes = min(max(1, pod_modes), coefficients.shape[1])
    states = np.asarray(coefficients[:, :n_modes], dtype=np.float64)
    times, states = sort_and_deduplicate(times, states)

    energy_capture = None
    if singular_values is not None:
        if "total_energy" in data.files:
            energy_denominator = float(np.asarray(data["total_energy"]).ravel()[0])
        else:
            energy_denominator = float(np.sum(singular_values**2))
        energy_capture = float(np.sum(singular_values[:n_modes] ** 2) / (energy_denominator + 1e-12))

    return ReducedDataset(
        name="pod",
        state_symbol="a",
        states=states,
        times=times,
        source=pod_path,
        metadata={
            "pod_modes_used": int(n_modes),
            "pod_total_modes": int(coefficients.shape[1]),
            "pod_energy_capture": energy_capture,
            "dt": infer_dt(times),
        },
    )


# Parâmetros principais
# Configuração atualizada para melhorar o caso POD:
# - 8 modos POD: aumenta energia capturada de ~72.8% para ~84.6%.
# - use_sine=False: remove termos trigonométricos que estavam deixando o modelo mais instável/denso.
# - varredura maior de alpha: permite comparar erro das derivadas x esparsidade.
config = SindyConfig(
    polynomial_order=2,
    use_sine=False,
    alpha_values=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    pod_modes=8,
    max_plot_states=6,
    heatmap_times=(0.001, 1.25, 2.50, 3.75, 5.00),
    heatmap_grid_size=64,
)


# Esta célula implementa o núcleo SINDy: biblioteca Theta, derivadas, regressão esparsa, equações e simulação.
# Função da biblioteca SINDy: gera todas as combinações de potências polinomiais até a ordem escolhida.
def generate_powers(n_variables: int, order: int) -> list[tuple[int, ...]]:
    if n_variables == 1:
        return [(power,) for power in range(order + 1)]
    powers: list[tuple[int, ...]] = []
    for power in range(order + 1):
        for suffix in generate_powers(n_variables - 1, order - power):
            powers.append((power,) + suffix)
    return sorted(powers, reverse=False)


# Classe da biblioteca SINDy: constrói e aplica a matriz Theta(X) com termos polinomiais e, opcionalmente, seno/cosseno.
@dataclass
class PolynomialLibrary:
    n_variables: int
    order: int
    use_sine: bool
    variable_symbol: str

    # Método de inicialização: pré-calcula as potências dos monômios e os nomes das features da biblioteca.
    def __post_init__(self) -> None:
        self.powers = generate_powers(self.n_variables, self.order)
        self.feature_names = self._build_feature_names()

    # Método auxiliar: cria nomes legíveis para cada termo da biblioteca, como 1, a1, a1*a2, sin(a1).
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

    # Método de transformação: avalia a biblioteca Theta para todos os estados reduzidos.
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

    # Método de transformação unitária: avalia Theta para apenas um estado, usado dentro da integração temporal.
    def transform_single(self, state: np.ndarray) -> np.ndarray:
        return self.transform(np.asarray(state, dtype=np.float64)[None, :])[0]


# Função de derivadas: calcula dX/dt numericamente a partir das coordenadas reduzidas e dos tempos.
def compute_derivatives(states: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.gradient(states, times, axis=0, edge_order=2 if len(times) > 2 else 1)


# Função de regressão esparsa: ajusta uma equação SINDy com Lasso para uma coordenada/derivada.
def fit_sparse_model(theta: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    scale = np.linalg.norm(theta, axis=0)
    scale[scale < 1e-12] = 1.0
    theta_scaled = theta / scale[None, :]
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000, tol=1e-5)
    model.fit(theta_scaled, target)
    return model.coef_ / scale


# Função principal SINDy: testa valores de alpha, descobre os coeficientes Xi das equações e escolhe o melhor ajuste de derivadas.
def discover_equations(theta: np.ndarray, derivatives: np.ndarray, alpha_values: tuple[float, ...]):
    print(f"   Library size: {theta.shape[1]} features")
    print(f"   Testing {len(alpha_values)} alpha values")

    # Número de equações a identificar: uma equação diferencial para cada coordenada reduzida.
    n_states = derivatives.shape[1]
    results = []
    # Para cada alpha, ajusta uma matriz Xi diferente e mede erro/esparsidade.
    for alpha in alpha_values:
        # Xi tem dimensão: número de termos da biblioteca x número de coordenadas.
        coefficients = np.zeros((theta.shape[1], n_states), dtype=np.float64)
        # Ajusta separadamente d(state_i)/dt = Theta @ Xi_i.
        for index in range(n_states):
            coefficients[:, index] = fit_sparse_model(theta, derivatives[:, index], alpha)
        # Depois de ajustar Xi, calcula as derivadas previstas pelo modelo.
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

    # Escolhe o alpha que melhor reproduz as derivadas numéricas.
    best_result = min(results, key=lambda item: item["derivative_l2_error"])
    print(
        f"   Selected alpha={best_result['alpha']:.4g} "
        f"with derivative L2={best_result['derivative_l2_error']:.6f}"
    )
    return best_result, results


# Função de simulação: integra no tempo o sistema dinâmico identificado pelo SINDy usando solve_ivp.
def simulate_sindy(
    library: PolynomialLibrary,
    coefficients: np.ndarray,
    times: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    # Função interna do integrador: calcula dX/dt = Theta(X) @ Xi no estado atual.
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


# Função de fallback: faz previsão de um passo usando o estado de referência anterior, útil quando a integração livre falha.
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


# Função de saída textual: transforma a matriz Xi em equações legíveis para cada coordenada reduzida.
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


# Esta célula contém funções que geram figuras diagnósticas no espaço reduzido.
# Função de gráfico: mostra como o erro das derivadas e o número de termos não nulos variam com alpha.
def plot_alpha_scan(scan_results: list[dict[str, Any]], output_dir: Path, title: str) -> None:
    force_matplotlib_agg()
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


# Função de gráfico: compara a trajetória reduzida de referência com a trajetória prevista pelo SINDy.
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
    force_matplotlib_agg()
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


# Função de gráfico: compara derivadas numéricas de referência com derivadas previstas pela biblioteca SINDy.
def plot_derivative_comparison(
    times: np.ndarray,
    derivatives: np.ndarray,
    derivatives_pred: np.ndarray,
    output_dir: Path,
    state_symbol: str,
    max_states: int,
) -> None:
    force_matplotlib_agg()
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


# Função de gráfico: gera mapa temporal das amplitudes das coordenadas reduzidas.
def plot_state_heatmap(
    times: np.ndarray,
    states: np.ndarray,
    output_dir: Path,
    state_symbol: str,
    title: str,
) -> None:
    force_matplotlib_agg()
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


# Esta célula contém funções para reconstrução física dos campos e heatmaps em tempos específicos.
# Função auxiliar de heatmaps: associa tempos solicitados aos snapshots disponíveis dentro de uma tolerância.
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


# Função auxiliar POD: interpola valores de células não estruturadas para uma grade regular 2D.
def interpolate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    points = np.column_stack([x, y])
    field = griddata(points, values, (grid_x, grid_y), method="linear")
    if np.isnan(field).any():
        nearest = griddata(points, values, (grid_x, grid_y), method="nearest")
        field = np.where(np.isnan(field), nearest, field)
    return np.asarray(field, dtype=np.float64)


# Função de reconstrução POD: reconstrói campos de velocidade físicos a partir dos coeficientes POD.
def build_pod_snapshot_reconstructor(dataset: ReducedDataset, grid_size: int):
    data = np.load(dataset.source, allow_pickle=True)
    basis = np.asarray(data["U"][:, : dataset.states.shape[1]], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64).ravel()
    y = np.asarray(data["y"], dtype=np.float64).ravel()
    mean_state = np.asarray(data["Xmean"], dtype=np.float64).ravel()
    centered = bool(np.asarray(data["centered"]).ravel()[0]) if "centered" in data.files else False
    n_cells = x.size

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_size),
        np.linspace(ymin, ymax, grid_size),
    )

    # Função interna de reconstrução: reconstrói um lote de estados reduzidos para campos físicos de velocidade.
    def reconstruct_batch(reduced_states: np.ndarray) -> np.ndarray:
        reduced_states = np.atleast_2d(np.asarray(reduced_states, dtype=np.float64))
        snapshots = []
        for state in reduced_states:
            vector = basis @ state
            if centered:
                vector = vector + mean_state
            ux = vector[:n_cells]
            uy = vector[n_cells : 2 * n_cells]
            ux_grid = interpolate_to_grid(x, y, ux, grid_x, grid_y)
            uy_grid = interpolate_to_grid(x, y, uy, grid_x, grid_y)
            snapshots.append(np.stack([ux_grid, uy_grid], axis=-1))
        return np.asarray(snapshots, dtype=np.float64)

    return reconstruct_batch


# Função de visualização de campo: salva Ux, Uy e magnitude da velocidade para um único snapshot.
def save_velocity_snapshot(frame: np.ndarray, output_path: Path, title: str) -> None:
    force_matplotlib_agg()
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


# Função de comparação de campo: salva referência, SINDy e erro para Ux, Uy e |U|.
def save_velocity_comparison(
    reference_frame: np.ndarray,
    predicted_frame: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    force_matplotlib_agg()
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


# Função de exemplos físicos: reconstrói e salva heatmaps nos tempos solicitados pelo usuário.
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

    reconstruct_batch = build_pod_snapshot_reconstructor(dataset, config.heatmap_grid_size)
    if matched:
        indices = [int(item["index"]) for item in matched]
        reference_frames = reconstruct_batch(dataset.states[indices])
        predicted_frames = reconstruct_batch(trajectory_pred[indices]) if trajectory_pred is not None else None

        for position, item in enumerate(matched):
            requested = float(item["requested_time"])
            actual = float(item["actual_time"])
            target_file = heatmap_dir / f"time_{requested:04.1f}s.png"
            if predicted_frames is None:
                title = f"POD reconstruction | requested={requested:.3f}s | used={actual:.3f}s"
                save_velocity_snapshot(reference_frames[position], target_file, title)
            else:
                field_error = relative_l2(reference_frames[position], predicted_frames[position])
                sindy_label = "SINDy" if trajectory_method is None else f"SINDy ({trajectory_method})"
                title = (
                    f"POD vs {sindy_label} | requested={requested:.3f}s | "
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
        print(f"   Missing exact times for POD: {missing}")
    return report


# Esta célula salva os resultados e organiza a execução completa do pipeline.
# Função de salvamento: grava equações, figuras, NPZ e summary.json com métricas e metadados.
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

    plot_state_heatmap(dataset.times, dataset.states, output_dir, dataset.state_symbol, "POD reduced coordinates")
    plot_alpha_scan(scan_results, output_dir, title="POD + SINDy alpha scan")
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
        "architecture": "POD+SINDy",
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


# Função orquestradora POD: executa em ordem carregamento, derivadas, biblioteca, identificação SINDy, simulação e salvamento.
def run_pod_analysis(
    case_dir: Path,
    config: SindyConfig | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    case_dir = Path(case_dir).resolve()
    output_root = case_dir / "SINDy" / "resultados" if output_root is None else Path(output_root)
    output_dir = output_root / "pod_sindy"
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = SindyConfig()

    # Passo 1: carregar coeficientes temporais POD que serão os estados reduzidos.
    dataset = load_pod_dataset(case_dir, config.pod_modes)

    print("\nProcessing POD + SINDy")
    print(f"   Source: {dataset.source}")
    print(f"   Reduced coordinates: {dataset.states.shape}")
    print(f"   Time range: [{dataset.times[0]:.6f}, {dataset.times[-1]:.6f}]")

    # Passo 2: calcular derivadas temporais dX/dt a partir dos dados reduzidos.
    derivatives = compute_derivatives(dataset.states, dataset.times)
    # Passo 3: construir a biblioteca candidata Theta(X), com termos polinomiais e seno/cosseno.
    library = PolynomialLibrary(
        n_variables=dataset.states.shape[1],
        order=config.polynomial_order,
        use_sine=config.use_sine,
        variable_symbol=dataset.state_symbol,
    )
    # Passo 4: avaliar a biblioteca nos dados para formar a matriz Theta.
    theta = library.transform(dataset.states)
    # Passo 5: descobrir as equações esparsas, isto é, os coeficientes Xi.
    best_result, scan_results = discover_equations(theta, derivatives, config.alpha_values)
    derivative_best_alpha = float(best_result["alpha"])

    for result in [best_result, *scan_results]:
        result["derivatives_reference"] = derivatives

    # Passo 6: selecionar inicialmente o melhor modelo pelas derivadas e tentar integração livre.
    selected_result = best_result
    trajectory_method = "solve_ivp"
    trajectory_pred, integration_message = simulate_sindy(
        library,
        selected_result["Xi"],
        dataset.times,
        dataset.states[0],
    )

    # Passo 7: se a integração livre falhar, usa previsão de um passo guiada pela referência.
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

    # Passo 8: salvar equações, figuras, métricas e arquivos NPZ/JSON.
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


# Bloco de busca automática POD+SINDy.
# Objetivo: encontrar uma configuração com erro aceitável e o menor número possível de termos.
import csv

# Função auxiliar: calcula uma pontuação combinando erro das derivadas, erro da trajetória e falta de esparsidade.
def pod_search_score(candidate: dict[str, Any], derivative_target: float, trajectory_target: float, sparsity_weight: float) -> float:
    derivative_part = candidate["derivative_l2_error"] / max(derivative_target, 1e-12)
    trajectory_part = candidate["trajectory_l2_error"] / max(trajectory_target, 1e-12)
    sparsity_penalty = 1.0 - candidate["sparsity"]
    return float(derivative_part + trajectory_part + sparsity_weight * sparsity_penalty)


# Função auxiliar: avalia uma configuração específica de POD+SINDy.
def evaluate_pod_candidate(
    case_dir: Path,
    pod_modes: int,
    polynomial_order: int,
    use_sine: bool,
    alpha: float,
    dataset_cache: dict[int, ReducedDataset],
) -> dict[str, Any]:
    # Passo 1: carrega ou reutiliza os coeficientes POD para o número de modos escolhido.
    if pod_modes not in dataset_cache:
        dataset_cache[pod_modes] = load_pod_dataset(case_dir, pod_modes)
    dataset = dataset_cache[pod_modes]

    # Passo 2: calcula as derivadas numéricas dos coeficientes POD.
    derivatives = compute_derivatives(dataset.states, dataset.times)

    # Passo 3: monta a biblioteca candidata Theta(a) para esta ordem e opção seno/cosseno.
    library = PolynomialLibrary(
        n_variables=dataset.states.shape[1],
        order=polynomial_order,
        use_sine=use_sine,
        variable_symbol=dataset.state_symbol,
    )
    theta = library.transform(dataset.states)

    # Passo 4: ajusta uma equação esparsa para cada derivada usando o alpha atual.
    coefficients = np.zeros((theta.shape[1], derivatives.shape[1]), dtype=np.float64)
    for state_index in range(derivatives.shape[1]):
        coefficients[:, state_index] = fit_sparse_model(theta, derivatives[:, state_index], alpha)

    # Passo 5: calcula erro das derivadas, isto é, compara dA/dt numérico com dA/dt previsto pelo SINDy.
    derivatives_pred = theta @ coefficients
    derivative_error = relative_l2(derivatives, derivatives_pred)

    # Passo 6: calcula trajetória de um passo guiada pela referência.
    # Isso é usado na busca porque é rápido e evita que uma configuração instável interrompa o teste.
    trajectory_pred = simulate_reference_guided_sindy(library, coefficients, dataset.times, dataset.states)
    trajectory_error = relative_l2(dataset.states, trajectory_pred)

    # Passo 7: mede esparsidade. Quanto mais perto de 1, mais esparso/simples é o modelo.
    n_possible_terms = int(coefficients.size)
    n_nonzero = int(np.count_nonzero(np.abs(coefficients) > 1e-8))
    sparsity = 1.0 - (n_nonzero / max(n_possible_terms, 1))

    return {
        "pod_modes": int(pod_modes),
        "pod_energy_capture": dataset.metadata.get("pod_energy_capture"),
        "polynomial_order": int(polynomial_order),
        "use_sine": bool(use_sine),
        "alpha": float(alpha),
        "library_size": int(theta.shape[1]),
        "n_possible_terms": n_possible_terms,
        "n_nonzero_terms": n_nonzero,
        "sparsity": float(sparsity),
        "derivative_l2_error": float(derivative_error),
        "trajectory_l2_error": float(trajectory_error),
        "trajectory_method": "reference_guided_one_step_search",
        "dataset": dataset,
        "library": library,
        "Xi": coefficients,
        "derivatives_reference": derivatives,
        "derivatives_pred": derivatives_pred,
        "trajectory_pred": trajectory_pred,
    }


# Função principal da busca: percorre todas as configurações e escolhe a mais esparsa dentro dos limites.
def run_pod_sparse_search(
    case_dir: Path,
    output_dir: Path,
    pod_modes_list: tuple[int, ...] = (4, 8, 16),
    polynomial_orders: tuple[int, ...] = (1, 2),
    use_sine_options: tuple[bool, ...] = (False, True),
    alpha_values: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    derivative_error_limit: float = 0.80,
    trajectory_error_limit: float = 0.005,
    min_energy_capture: float = 0.80,
    sparsity_weight: float = 0.25,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache: dict[int, ReducedDataset] = {}
    candidates: list[dict[str, Any]] = []

    print("\nIniciando busca POD+SINDy")
    print(f"   pod_modes_list      = {pod_modes_list}")
    print(f"   polynomial_orders   = {polynomial_orders}")
    print(f"   use_sine_options    = {use_sine_options}")
    print(f"   alpha_values        = {alpha_values}")
    print(f"   derivative_limit    = {derivative_error_limit}")
    print(f"   trajectory_limit    = {trajectory_error_limit}")
    print(f"   min_energy_capture  = {min_energy_capture}")

    # Passo de varredura: testa todas as combinações uma por uma.
    for pod_modes in pod_modes_list:
        for polynomial_order in polynomial_orders:
            for use_sine in use_sine_options:
                for alpha in alpha_values:
                    candidate = evaluate_pod_candidate(
                        case_dir=case_dir,
                        pod_modes=pod_modes,
                        polynomial_order=polynomial_order,
                        use_sine=use_sine,
                        alpha=alpha,
                        dataset_cache=dataset_cache,
                    )
                    candidate["accepted"] = bool(
                        candidate["derivative_l2_error"] <= derivative_error_limit
                        and candidate["trajectory_l2_error"] <= trajectory_error_limit
                        and (candidate["pod_energy_capture"] is None or candidate["pod_energy_capture"] >= min_energy_capture)
                    )
                    candidate["score"] = pod_search_score(candidate, derivative_error_limit, trajectory_error_limit, sparsity_weight)
                    candidates.append(candidate)

                    status = "OK" if candidate["accepted"] else "--"
                    print(
                        f"[{status}] modes={pod_modes:2d} | order={polynomial_order} | sine={use_sine} | "
                        f"alpha={alpha:.1e} | deriv={candidate['derivative_l2_error']:.4f} | "
                        f"traj={candidate['trajectory_l2_error']:.5f} | "
                        f"sparsity={candidate['sparsity']:.3f} | nnz={candidate['n_nonzero_terms']}"
                    )

    # Seleção: primeiro tenta pegar o aceitável mais esparso; se não houver, usa melhor score.
    accepted = [item for item in candidates if item["accepted"]]
    if accepted:
        best = sorted(
            accepted,
            key=lambda item: (-item["sparsity"], item["derivative_l2_error"], item["trajectory_l2_error"]),
        )[0]
        selection_reason = "modelo aceitável com máxima esparsidade"
    else:
        best = sorted(candidates, key=lambda item: item["score"])[0]
        selection_reason = "nenhum modelo atendeu aos limites; escolhido melhor compromisso por score"

    # Teste final: tenta integração livre apenas para o melhor candidato.
    free_trajectory, integration_message = simulate_sindy(
        best["library"],
        best["Xi"],
        best["dataset"].times,
        best["dataset"].states[0],
    )
    if free_trajectory is None:
        free_trajectory_error = None
        free_integration_success = False
    else:
        free_trajectory_error = relative_l2(best["dataset"].states, free_trajectory)
        free_integration_success = True

    # Salva tabela CSV sem arrays para facilitar leitura.
    csv_path = output_dir / "pod_sindy_search_results.csv"
    csv_fields = [
        "accepted", "score", "pod_modes", "pod_energy_capture", "polynomial_order", "use_sine", "alpha",
        "library_size", "n_possible_terms", "n_nonzero_terms", "sparsity",
        "derivative_l2_error", "trajectory_l2_error", "trajectory_method",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow({field: item[field] for field in csv_fields})

    # Salva o melhor modelo em NPZ.
    np.savez_compressed(
        output_dir / "best_pod_sindy_model.npz",
        times=best["dataset"].times,
        states=best["dataset"].states,
        Xi=best["Xi"],
        feature_names=np.asarray(best["library"].feature_names, dtype=object),
        derivatives_reference=best["derivatives_reference"],
        derivatives_pred=best["derivatives_pred"],
        trajectory_pred=best["trajectory_pred"],
        free_trajectory_pred=np.full_like(best["dataset"].states, np.nan) if free_trajectory is None else free_trajectory,
    )

    # Salva as equações do melhor modelo.
    equations = format_equations(best["Xi"], best["library"].feature_names, best["dataset"].state_symbol)
    (output_dir / "best_equations.txt").write_text(equations, encoding="utf-8")

    summary = {
        "selection_reason": selection_reason,
        "n_tested": len(candidates),
        "n_accepted": len(accepted),
        "limits": {
            "derivative_error_limit": derivative_error_limit,
            "trajectory_error_limit": trajectory_error_limit,
            "min_energy_capture": min_energy_capture,
        },
        "best": {
            key: make_json_ready(value)
            for key, value in best.items()
            if key not in {"dataset", "library", "Xi", "derivatives_reference", "derivatives_pred", "trajectory_pred"}
        },
        "free_integration": {
            "success": free_integration_success,
            "trajectory_l2_error": None if free_trajectory_error is None else float(free_trajectory_error),
            "message": integration_message,
        },
        "files": {
            "csv": str(csv_path),
            "best_model": str(output_dir / "best_pod_sindy_model.npz"),
            "best_equations": str(output_dir / "best_equations.txt"),
        },
    }
    (output_dir / "pod_sindy_search_summary.json").write_text(json.dumps(make_json_ready(summary), indent=2), encoding="utf-8")

    print("\nMelhor configuração encontrada:")
    print(json.dumps(make_json_ready(summary["best"]), indent=2, ensure_ascii=False))
    print("\nTeste de integração livre do melhor modelo:")
    print(json.dumps(make_json_ready(summary["free_integration"]), indent=2, ensure_ascii=False))
    print(f"\nResultados da busca salvos em: {output_dir}")
    return summary


# Execute esta célula para rodar a busca automática.
# Pode demorar, principalmente quando pod_modes_list inclui 16 ou mais modos.
search_output_dir = OUTPUT_ROOT / "pod_sindy_search_auto"

search_summary = run_pod_sparse_search(
    case_dir=CASE_DIR,
    output_dir=search_output_dir,
    pod_modes_list=(4, 8, 16),
    polynomial_orders=(1, 2),
    use_sine_options=(False, True),
    alpha_values=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    derivative_error_limit=0.80,
    trajectory_error_limit=0.005,
    min_energy_capture=0.80,
    sparsity_weight=0.25,
)
search_summary


# Esta célula executa o pipeline completo com a configuração definida acima.
# Reforça o backend Agg caso o kernel já tenha carregado matplotlib_inline antes.
plt.switch_backend("Agg")
summary = run_pod_analysis(CASE_DIR, config, OUTPUT_ROOT)
summary


# Esta célula imprime o resumo final em JSON para inspeção rápida das métricas e arquivos gerados.
print(json.dumps(summary, indent=2, ensure_ascii=False))
