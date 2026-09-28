"""
ConvSAE (Convolutional Sparse Autoencoder) para cavity_refinado
(MESMAS MELHORIAS DO damBreakLaminarFine)

- OpenFOAM -> grade 2D (imagem) via interpolação (griddata)
- Split temporal (train/val)
- Normalização por canal (apenas no treino) + clipping robusto
- ConvSAE com BatchNorm + SpatialDropout2D + esparsidade (L1 activity no latente)
- Loss robusta (Huber) opcional
- Callbacks (EarlyStopping + ReduceLROnPlateau + Checkpoint)
- Diagnóstico:
  (1) loss/val_loss
  (2) generalization gap
  (3) overfit score (gap/|val|)
  (4) MSE por snapshot (train vs val)
  (5) relL2 por snapshot (train vs val)
  (6) histogramas relL2 train/val
  (7) métricas por canal (Ux, Uy)
  (8) outliers (topK) com imagens (true | recon | err)
  (9) relL2 no físico (desnormalizado) + outliers físicos
- Salva:
  modelo, melhor modelo, encoder, normalização, latentes z(t),
  métricas npz, nan_stats e figuras

Como rodar:
cd cavity_refinado/SAE/src
python ConvAE_analysis_Matriz.py
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from fluidfoam import readmesh, readfield
from scipy.interpolate import griddata
from tensorflow.keras import regularizers


# ============================================================
# CONFIG (AJUSTE AQUI)
# ============================================================

# Este arquivo deve ficar em: cavity_refinado/SAE/src/
THIS_DIR = Path(__file__).resolve().parent
CASE_DIR = THIS_DIR.parent.parent  # -> cavity_refinado/

# validação do caminho do caso
poly = CASE_DIR / "constant" / "polyMesh"
if not poly.exists():
    raise FileNotFoundError(
        f"CASE_DIR errado: não achei {poly}\n"
        f"Você colocou CASE_DIR={CASE_DIR.resolve()}\n"
        f"Ele precisa apontar para a pasta do caso (onde existe constant/, system/, 0/, ...)."
    )

print("✅ CASE_DIR OK:", CASE_DIR.resolve())

OUT_DIR = CASE_DIR / "SAE" / "resultados" / "ConvAE"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("✅ Salvando em:", OUT_DIR.resolve())

# grade 2D (imagem)
NX, NY = 128, 128
N_TIMESTEPS = None
INTERP_METHOD = "linear"   # "linear" recomendado; "cubic" pode gerar NaN demais
TRAIN_RATIO = 0.8          # 0.8 / 0.85 costuma ser melhor se poucos timesteps
PADDING = 0.02             # padding no domínio (para evitar bordas apertadas)

# treino
LATENT_CH = 16
LR = 1e-3
EPOCHS = 200
BATCH = 4
SHUFFLE = False            # ✅ série temporal → não embaralhar

# regularização / robustez
L1_LATENT = 1e-6           # esparsidade no latente (tente 1e-6 a 1e-4)
SPATIAL_DROPOUT = 0.10     # regularização leve
CLIP_ZSCORE = 6.0          # clipping após normalizar (robusto para outliers)
USE_HUBER = True
HUBER_DELTA = 1.0

# callbacks
EARLY_PATIENCE = 20
PLATEAU_PATIENCE = 8

# diagnóstico
OUTLIER_TOPK = 5
SAVE_OUTLIER_IMAGES = True
SEED = 123

TAG = "convsae"


# ============================================================
# 0) Reprodutibilidade
# ============================================================
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 1) OpenFOAM -> Grade 2D (imagem)
# ============================================================
def list_time_dirs(case_dir: str, n_timesteps=None):
    time_dirs = sorted(
        [
            d for d in os.listdir(case_dir)
            if os.path.isdir(os.path.join(case_dir, d))
            and d.replace(".", "").replace("-", "").isdigit()
        ],
        key=lambda s: float(s),
    )
    if n_timesteps:
        time_dirs = time_dirs[:n_timesteps]
    return time_dirs


def build_regular_grid(xc, yc, nx=128, ny=128, padding=0.02):
    xmin, xmax = xc.min(), xc.max()
    ymin, ymax = yc.min(), yc.max()
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= padding * dx
    xmax += padding * dx
    ymin -= padding * dy
    ymax += padding * dy
    xg = np.linspace(xmin, xmax, nx)
    yg = np.linspace(ymin, ymax, ny)
    Xg, Yg = np.meshgrid(xg, yg)
    return Xg, Yg


def read_U_components(case_dir: str, time_dir: str, nCells: int):
    """
    Lê U e retorna (Ux, Uy) com tamanho nCells.
    Robusto para:
      - campo uniforme: shape (3,1)
      - (3, nCells)
      - (nCells, 3)
      - vetor achatado 3*nCells
    """
    Uf = np.asarray(readfield(case_dir, time_dir, "U"))

    # (3,1) -> campo uniforme
    if Uf.ndim == 2 and Uf.shape == (3, 1):
        Ux = np.full(nCells, float(Uf[0, 0]), dtype=float)
        Uy = np.full(nCells, float(Uf[1, 0]), dtype=float)
        return Ux, Uy

    # (3, nCells)
    if Uf.ndim == 2 and Uf.shape[0] == 3 and Uf.shape[1] == nCells:
        return Uf[0, :].astype(float), Uf[1, :].astype(float)

    # (nCells, 3)
    if Uf.ndim == 2 and Uf.shape[0] == nCells and Uf.shape[1] >= 2:
        return Uf[:, 0].astype(float), Uf[:, 1].astype(float)

    # achatado: (3*nCells,)
    if Uf.ndim == 1 and Uf.size == 3 * nCells:
        Uf2 = Uf.reshape(nCells, 3)
        return Uf2[:, 0].astype(float), Uf2[:, 1].astype(float)

    raise ValueError(f"Formato inesperado do U em {time_dir}: Uf.shape={Uf.shape} (nCells={nCells})")


def load_openfoam_as_grids(case_dir: str, nx=128, ny=128, n_timesteps=None, method="linear", padding=0.02):
    """
    Retorna:
      X: (nTimes, ny, nx, 2) canais: [Ux, Uy]
      t: (nTimes,)
      nan_stats: dict com percentuais de NaN antes do nearest-fill
    """
    # centros de célula
    xc, yc, _zc = readmesh(case_dir)
    xc = np.asarray(xc).ravel()
    yc = np.asarray(yc).ravel()
    nCells = xc.size

    time_dirs = list_time_dirs(case_dir, n_timesteps=n_timesteps)

    Xg, Yg = build_regular_grid(xc, yc, nx=nx, ny=ny, padding=padding)
    pts = np.column_stack([xc, yc])     # (nCells, 2)
    grid_pts = (Xg, Yg)

    X_list, t_list = [], []
    nan_fill_ratio = []  # fração média de NaN antes do nearest-fill

    print(f"📂 Timesteps encontrados: {len(time_dirs)}")
    for td in time_dirs:
        try:
            Ux, Uy = read_U_components(case_dir, td, nCells=nCells)

            Uxg = griddata(pts, Ux, grid_pts, method=method)
            Uyg = griddata(pts, Uy, grid_pts, method=method)

            nan_before = 0.5 * (np.mean(np.isnan(Uxg)) + np.mean(np.isnan(Uyg)))
            nan_fill_ratio.append(float(nan_before))

            # completa NaNs fora do convexo com nearest
            if np.isnan(Uxg).any() or np.isnan(Uyg).any():
                Uxg_near = griddata(pts, Ux, grid_pts, method="nearest")
                Uyg_near = griddata(pts, Uy, grid_pts, method="nearest")
                Uxg = np.where(np.isnan(Uxg), Uxg_near, Uxg)
                Uyg = np.where(np.isnan(Uyg), Uyg_near, Uyg)

            frame = np.stack([Uxg, Uyg], axis=-1).astype(np.float32)  # (ny,nx,2)
            X_list.append(frame)
            t_list.append(float(td))

            print(f"   ✓ t={td:>10s} | nan_ratio(before fill)={nan_before:.4f}")

        except Exception as e:
            print(f"   ⚠️ erro no timestep {td}: {e}")
            continue

    X = np.array(X_list, dtype=np.float32)
    t = np.array(t_list, dtype=np.float32)

    print(f"\n✓ X grids shape = {X.shape} (nTimes, ny, nx, 2)")
    print(f"✓ t shape       = {t.shape}")

    nan_stats = {
        "nan_ratio_mean": float(np.mean(nan_fill_ratio)) if len(nan_fill_ratio) else None,
        "nan_ratio_max": float(np.max(nan_fill_ratio)) if len(nan_fill_ratio) else None,
        "nan_ratio_per_timestep": np.array(nan_fill_ratio, dtype=np.float32),
    }
    return X, t, nan_stats


def time_split(X, t, train_ratio=0.8):
    idx = np.argsort(t)
    X = X[idx]
    t = t[idx]
    n = len(t)
    n_train = int(train_ratio * n)
    return X[:n_train], X[n_train:], t[:n_train], t[n_train:]


def normalize_train_only(X_train, X_val, eps=1e-12):
    # estatísticas por canal (Ux, Uy), mantendo estrutura espacial
    mu = X_train.mean(axis=(0, 1, 2), keepdims=True)          # (1,1,1,C)
    sig = X_train.std(axis=(0, 1, 2), keepdims=True) + eps    # (1,1,1,C)
    return (X_train - mu) / sig, (X_val - mu) / sig, mu, sig


# ============================================================
# 2) ConvSAE + Esparsidade no latente
# ============================================================
def build_conv_sae(ny, nx, channels=2, latent_channels=16, l1_latent=1e-6, spatial_dropout=0.1):
    inp = tf.keras.Input(shape=(ny, nx, channels), name="x")

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)  # /2

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)  # /4

    if spatial_dropout and spatial_dropout > 0:
        x = tf.keras.layers.SpatialDropout2D(spatial_dropout)(x)

    # Latente com regularização de esparsidade (L1 na atividade)
    z = tf.keras.layers.Conv2D(
        latent_channels, 3, padding="same", activation="relu",
        activity_regularizer=regularizers.l1(l1_latent),
        name="latent"
    )(x)

    # Decoder
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(z)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D(2)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D(2)(x)

    out = tf.keras.layers.Conv2D(channels, 3, padding="same", activation=None, name="x_hat")(x)

    return tf.keras.Model(inp, out, name="ConvSAE")


# ============================================================
# 3) Métricas e Diagnóstico
# ============================================================
def flatten_batch(X):
    return X.reshape((X.shape[0], -1))


def mse_per_sample(Xhat, X):
    Xh = flatten_batch(Xhat)
    Xf = flatten_batch(X)
    return np.mean((Xh - Xf) ** 2, axis=1)


def rel_l2_per_sample(Xhat, X, eps=1e-12):
    Xh = flatten_batch(Xhat)
    Xf = flatten_batch(X)
    num = np.linalg.norm(Xh - Xf, axis=1)
    den = np.linalg.norm(Xf, axis=1) + eps
    return num / den


def global_r2(Xhat, X, eps=1e-12):
    Xh = flatten_batch(Xhat).ravel()
    Xf = flatten_batch(X).ravel()
    ss_res = np.sum((Xf - Xh) ** 2)
    ss_tot = np.sum((Xf - Xf.mean()) ** 2) + eps
    return 1.0 - ss_res / ss_tot


def channel_metrics(Xhat, X):
    # X: (N,H,W,C)
    if X.ndim < 4:
        return None
    C = X.shape[-1]
    out = {"mse": [], "relL2": []}
    for c in range(C):
        xc = X[..., c]
        xhc = Xhat[..., c]
        out["mse"].append(np.mean((flatten_batch(xhc) - flatten_batch(xc)) ** 2))
        out["relL2"].append(np.mean(rel_l2_per_sample(xhc, xc)))
    out["mse"] = np.array(out["mse"])
    out["relL2"] = np.array(out["relL2"])
    return out


def save_recon_compare(X, Xhat, outpath, title=""):
    """
    Salva comparação de reconstrução:
    linha 1: Ux (true, recon, err)
    linha 2: Uy (true, recon, err)
    """
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for c, name in enumerate(["Ux", "Uy"]):
        ax[c, 0].imshow(X[..., c])
        ax[c, 0].set_title(f"{name} true")
        ax[c, 1].imshow(Xhat[..., c])
        ax[c, 1].set_title(f"{name} recon")
        ax[c, 2].imshow(Xhat[..., c] - X[..., c])
        ax[c, 2].set_title(f"{name} err")
        for j in range(3):
            ax[c, j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def analyze_training_errors(
    model, history,
    X_train, X_val,
    t_train=None, t_val=None,
    out_dir="resultados", tag="convsae",
    denorm=None, outlier_topk=5,
    save_outlier_images=True
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outlier_dir = out_dir / "outliers"
    if save_outlier_images:
        outlier_dir.mkdir(parents=True, exist_ok=True)

    loss = np.array(history.history.get("loss", []), dtype=float)
    vloss = np.array(history.history.get("val_loss", []), dtype=float)
    epochs = np.arange(1, len(loss) + 1)

    gap = vloss - loss
    gap_rel = gap / (np.abs(vloss) + 1e-12)

    # (1) curvas
    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, loss, label="train")
    plt.plot(epochs, vloss, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs epoch")
    plt.grid(True, alpha=0.3); plt.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_01_loss.png", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, gap, label="val - train")
    plt.xlabel("epoch"); plt.ylabel("gap"); plt.title("Generalization gap")
    plt.grid(True, alpha=0.3); plt.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_02_gap.png", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, gap_rel, label="gap / |val|")
    plt.xlabel("epoch"); plt.ylabel("relative gap"); plt.title("Overfitting score (relative gap)")
    plt.grid(True, alpha=0.3); plt.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_03_overfit_score.png", dpi=150); plt.close(fig)

    best_ep = int(np.argmin(vloss)) + 1 if len(vloss) else None

    # (2) reconstruções
    X_train_hat = model.predict(X_train, verbose=0)
    X_val_hat   = model.predict(X_val,   verbose=0)

    tr_mse = mse_per_sample(X_train_hat, X_train)
    va_mse = mse_per_sample(X_val_hat,   X_val)
    tr_rel = rel_l2_per_sample(X_train_hat, X_train)
    va_rel = rel_l2_per_sample(X_val_hat,   X_val)

    tr_r2 = global_r2(X_train_hat, X_train)
    va_r2 = global_r2(X_val_hat,   X_val)

    ch_tr = channel_metrics(X_train_hat, X_train)
    ch_va = channel_metrics(X_val_hat,   X_val)

    def plot_by_time(y_train, y_val, ylabel, title, fname):
        fig = plt.figure(figsize=(10, 4))
        if t_train is None:
            plt.plot(np.arange(len(y_train)), y_train, "o-", label="train")
        else:
            plt.plot(t_train, y_train, "o-", label="train")
        if t_val is None:
            plt.plot(np.arange(len(y_val)) + len(y_train), y_val, "o-", label="val")
        else:
            plt.plot(t_val, y_val, "o-", label="val")
        plt.xlabel("tempo" if (t_train is not None or t_val is not None) else "index")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3); plt.legend()
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=150); plt.close(fig)

    plot_by_time(tr_mse, va_mse, "MSE", "MSE por snapshot (train vs val)", f"{tag}_04_mse_snapshot.png")
    plot_by_time(tr_rel, va_rel, "relL2", "Erro relativo L2 por snapshot (train vs val)", f"{tag}_05_relL2_snapshot.png")

    # (3) histogramas
    def hist_plot(values, title, fname):
        fig = plt.figure(figsize=(8, 4))
        plt.hist(values, bins=30)
        plt.title(title); plt.xlabel("valor"); plt.ylabel("contagem")
        plt.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=150); plt.close(fig)

    hist_plot(tr_rel, "Hist relL2 (train)", f"{tag}_06_hist_relL2_train.png")
    hist_plot(va_rel, "Hist relL2 (val)",   f"{tag}_07_hist_relL2_val.png")

    # (4) por canal
    if ch_tr is not None:
        C = len(ch_tr["mse"])
        labels = ["Ux", "Uy"][:C]

        fig = plt.figure(figsize=(8, 4))
        plt.plot(range(C), ch_tr["mse"], "o-", label="train")
        plt.plot(range(C), ch_va["mse"], "o-", label="val")
        plt.xticks(range(C), labels)
        plt.xlabel("canal"); plt.ylabel("MSE"); plt.title("MSE por canal")
        plt.grid(True, alpha=0.3); plt.legend()
        fig.tight_layout(); fig.savefig(out_dir / f"{tag}_08_mse_by_channel.png", dpi=150); plt.close(fig)

        fig = plt.figure(figsize=(8, 4))
        plt.plot(range(C), ch_tr["relL2"], "o-", label="train")
        plt.plot(range(C), ch_va["relL2"], "o-", label="val")
        plt.xticks(range(C), labels)
        plt.xlabel("canal"); plt.ylabel("relL2"); plt.title("relL2 por canal")
        plt.grid(True, alpha=0.3); plt.legend()
        fig.tight_layout(); fig.savefig(out_dir / f"{tag}_09_relL2_by_channel.png", dpi=150); plt.close(fig)

    # (5) outliers
    def topk_indices(x, k):
        k = min(k, len(x))
        return np.argsort(-x)[:k]

    tr_bad = topk_indices(tr_rel, outlier_topk)
    va_bad = topk_indices(va_rel, outlier_topk)

    if save_outlier_images:
        for rank, i in enumerate(va_bad, start=1):
            tinfo = float(t_val[i]) if t_val is not None else None
            title = f"VAL outlier #{rank} | idx={int(i)} | t={tinfo} | relL2={float(va_rel[i]):.4f}"
            save_recon_compare(
                X_val[i], X_val_hat[i],
                outlier_dir / f"{tag}_val_outlier_{rank:02d}_idx{int(i)}.png",
                title=title
            )

    # (6) desnormalizado opcional (físico)
    phys = None
    if denorm is not None and ("mu" in denorm) and ("sig" in denorm):
        mu = denorm["mu"]; sig = denorm["sig"]
        X_train_phys = X_train * sig + mu
        X_val_phys   = X_val   * sig + mu
        X_train_hat_phys = X_train_hat * sig + mu
        X_val_hat_phys   = X_val_hat   * sig + mu

        tr_rel_phys = rel_l2_per_sample(X_train_hat_phys, X_train_phys)
        va_rel_phys = rel_l2_per_sample(X_val_hat_phys,   X_val_phys)

        plot_by_time(tr_rel_phys, va_rel_phys, "relL2 (físico)", "Erro relL2 (desnormalizado)", f"{tag}_10_relL2_phys.png")
        phys = {"train_relL2_phys": tr_rel_phys, "val_relL2_phys": va_rel_phys}

        if save_outlier_images:
            for rank, i in enumerate(va_bad, start=1):
                tinfo = float(t_val[i]) if t_val is not None else None
                title = f"VAL outlier (PHYS) #{rank} | idx={int(i)} | t={tinfo} | relL2_phys={float(va_rel_phys[i]):.4f}"
                save_recon_compare(
                    X_val_phys[i], X_val_hat_phys[i],
                    outlier_dir / f"{tag}_val_outlier_phys_{rank:02d}_idx{int(i)}.png",
                    title=title
                )

    # salvar npz
    np.savez_compressed(
        out_dir / f"{tag}_metrics.npz",
        loss=loss, val_loss=vloss, gap=gap, gap_rel=gap_rel,
        train_mse=tr_mse, val_mse=va_mse,
        train_relL2=tr_rel, val_relL2=va_rel,
        train_r2=float(tr_r2), val_r2=float(va_r2),
        train_bad_idx=tr_bad, val_bad_idx=va_bad,
        **({"train_relL2_phys": phys["train_relL2_phys"], "val_relL2_phys": phys["val_relL2_phys"]} if phys else {})
    )

    print("\n======= DIAGNÓSTICO =======")
    print(f"Melhor época (menor val_loss): {best_ep}")
    print(f"R² global: train={tr_r2:.4f} | val={va_r2:.4f}")
    print("Piores (val) relL2:", [(int(i), float(t_val[i]) if t_val is not None else None, float(va_rel[i])) for i in va_bad])
    print("Arquivos salvos em:", out_dir.resolve())
    print("===========================\n")


# ============================================================
# 4) PIPELINE ÚNICO (treino + diagnóstico + salvamento)
# ============================================================
def run_convSAE_pipeline():
    print("\n" + "=" * 80)
    print("ConvSAE PIPELINE - cavity_refinado".center(80))
    print("=" * 80)

    # 1) carregar e interpolar
    X, t, nan_stats = load_openfoam_as_grids(
        str(CASE_DIR),
        nx=NX, ny=NY,
        n_timesteps=N_TIMESTEPS,
        method=INTERP_METHOD,
        padding=PADDING
    )

    if len(t) < 4:
        raise RuntimeError("Poucos timesteps válidos. Verifique leitura/interpolação.")

    print("\n--- NaN stats (antes do nearest fill) ---")
    print(f"nan_ratio_mean: {nan_stats['nan_ratio_mean']}")
    print(f"nan_ratio_max : {nan_stats['nan_ratio_max']}")
    np.savez_compressed(OUT_DIR / "nan_stats.npz", **nan_stats, t=t)

    # 2) split temporal
    X_train, X_val, t_train, t_val = time_split(X, t, train_ratio=TRAIN_RATIO)

    # 3) normalização por canal (só treino)
    X_train_n, X_val_n, mu, sig = normalize_train_only(X_train, X_val)

    # 4) clipping robusto após normalizar
    if CLIP_ZSCORE is not None:
        X_train_n = np.clip(X_train_n, -CLIP_ZSCORE, CLIP_ZSCORE)
        X_val_n   = np.clip(X_val_n,   -CLIP_ZSCORE, CLIP_ZSCORE)

    # salvar normalização
    np.savez_compressed(OUT_DIR / f"{TAG}_norm.npz", mu=mu, sig=sig)

    # 5) construir ConvSAE
    model = build_conv_sae(
        NY, NX, channels=2,
        latent_channels=LATENT_CH,
        l1_latent=L1_LATENT,
        spatial_dropout=SPATIAL_DROPOUT
    )

    # loss
    if USE_HUBER:
        loss_fn = tf.keras.losses.Huber(delta=HUBER_DELTA)
    else:
        loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=loss_fn
    )

    # callbacks
    ckpt_best = OUT_DIR / f"{TAG}_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=PLATEAU_PATIENCE,
            factor=0.5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False
        )
    ]

    print("\n--- Modelo ---")
    model.summary()

    # 6) treinar
    history = model.fit(
        X_train_n, X_train_n,
        validation_data=(X_val_n, X_val_n),
        epochs=EPOCHS,
        batch_size=BATCH,
        shuffle=SHUFFLE,
        callbacks=callbacks,
        verbose=2
    )

    # salvar modelo final
    model.save(OUT_DIR / f"{TAG}.keras")

    # salvar encoder
    encoder = tf.keras.Model(model.input, model.get_layer("latent").output, name="encoder")
    encoder.save(OUT_DIR / "encoder.keras")

    # 7) latentes z(t)
    z_train = encoder.predict(X_train_n, verbose=0)
    z_val   = encoder.predict(X_val_n,   verbose=0)
    np.savez_compressed(
        OUT_DIR / "latents.npz",
        t_train=t_train, t_val=t_val,
        z_train=z_train, z_val=z_val
    )

    # 8) diagnóstico (salva figs + métricas)
    analyze_training_errors(
        model=model,
        history=history,
        X_train=X_train_n, X_val=X_val_n,
        t_train=t_train, t_val=t_val,
        out_dir=OUT_DIR,
        tag=TAG,
        denorm={"mu": mu, "sig": sig},
        outlier_topk=OUTLIER_TOPK,
        save_outlier_images=SAVE_OUTLIER_IMAGES
    )

    print("\n✅ FIM. Confira:", OUT_DIR.resolve())


if __name__ == "__main__":
    run_convSAE_pipeline()