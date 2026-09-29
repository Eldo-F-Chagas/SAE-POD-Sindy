#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparação modular dos resultados POD, SAE, SINDy e SAE--POD--SINDy.

Este código usa somente dados salvos pelas simulações em OpenFOAM/SAE-POD-Sindy/run.
Gera tabelas CSV e figuras comparativas na pasta ComparacaoResultados.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/home/mlep-pc-ubuntu/OpenFOAM/SAE-POD-Sindy")
RUN = BASE / "run"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

CASOS = {
    "Laminar": {
        "hibrido": RUN / "damBreakLaminar 10 000/Codigos damBreakLaminar 10.000/SINDy/resultados/sae_pod_sindy/order1_time_alasso",
        "pod_sindy": RUN / "damBreakLaminar 10 000/Codigos damBreakLaminar 10.000/SINDy/resultados/pod_sindy/summary.json",
        "sae_sindy": RUN / "damBreakLaminar 10 000/Codigos damBreakLaminar 10.000/SINDy/resultados/sae_sindy/summary.json",
        "sae": RUN / "damBreakLaminar 10 000/Codigos damBreakLaminar 10.000/SAE/resultados/sae_dense_metrics.npz",
        "pod": RUN / "damBreakLaminar 10 000/Codigos damBreakLaminar 10.000/POD/resultados/resultados_pod_completos.npz",
    },
    "URANS": {
        "hibrido": RUN / "damBreakLaminar(URANS)10.000/Codigos(damBreakLaminar10.00(URANS))/SINDy/resultados/sae_pod_sindy/order1_time_alasso",
        "pod_sindy": RUN / "damBreakLaminar(URANS)10.000/Codigos(damBreakLaminar10.00(URANS))/SINDy/resultados/pod_sindy/summary.json",
        "sae_sindy": RUN / "damBreakLaminar(URANS)10.000/Codigos(damBreakLaminar10.00(URANS))/SINDy/resultados/sae_sindy/summary.json",
        "sae": RUN / "damBreakLaminar(URANS)10.000/Codigos(damBreakLaminar10.00(URANS))/SAE/resultados/sae_dense_metrics.npz",
        "pod": RUN / "damBreakLaminar(URANS)10.000/Codigos(damBreakLaminar10.00(URANS))/POD/resultados/resultados_pod_completos.npz",
    },
    "LES--WALE": {
        "hibrido": RUN / "damBreakLaminar(LES_WALE)10.000/Codigos damBreakLaminar LES_WALE 10.000/SINDy/resultados/sae_pod_sindy/order1_time_alasso",
        "pod_sindy": RUN / "damBreakLaminar(LES_WALE)10.000/Codigos damBreakLaminar LES_WALE 10.000/SINDy/resultados/pod_sindy/summary.json",
        "sae_sindy": RUN / "damBreakLaminar(LES_WALE)10.000/Codigos damBreakLaminar LES_WALE 10.000/SINDy/resultados/sae_sindy/summary.json",
        "sae": RUN / "damBreakLaminar(LES_WALE)10.000/Codigos damBreakLaminar LES_WALE 10.000/SAE/resultados/sae_dense_metrics.npz",
        "pod": RUN / "damBreakLaminar(LES_WALE)10.000/Codigos damBreakLaminar LES_WALE 10.000/POD/resultados/resultados_pod_completos.npz",
    },
    "Cavity": {
        "hibrido": RUN / "cavity 10 000/Codigos Cavity 10.000/SINDy/resultados/sae_pod_sindy/order1_time_alasso",
        "pod_sindy": RUN / "cavity 10 000/Codigos Cavity 10.000/SINDy/resultados_corrigidos/pod_sindy/summary.json",
        "sae_sindy": RUN / "cavity 10 000/Codigos Cavity 10.000/SINDy/resultados/sae_sindy/summary.json",
        "sae": RUN / "cavity 10 000/Codigos Cavity 10.000/SAE/resultados/sae_dense_metrics.npz",
        "pod": RUN / "cavity 10 000/Codigos Cavity 10.000/POD/resultados_corrigidos/pod_results.npz",
        "pod_summary": RUN / "cavity 10 000/Codigos Cavity 10.000/POD/resultados_corrigidos/summary.json",
    },
}


def safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(x: Any, default=np.nan) -> float:
    try:
        a = np.asarray(x)
        if a.size == 0:
            return default
        return float(a.reshape(-1)[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return default


def rel_l2(y: np.ndarray, yhat: np.ndarray, axis=None):
    return np.linalg.norm(yhat - y, axis=axis) / np.maximum(np.linalg.norm(y, axis=axis), 1e-14)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ssr = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y, axis=0)) ** 2))
    return 1.0 - ssr / max(sst, 1e-14)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frow(row: dict[str, str], key: str, default=np.nan) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def resumo_sae(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    val_rel = np.asarray(z.get("val_relL2", []), float)
    train_rel = np.asarray(z.get("train_relL2", []), float)
    val_loss = np.asarray(z.get("val_loss", []), float)
    return {
        "sae_train_r2": scalar(z.get("train_r2", np.nan)),
        "sae_val_r2": scalar(z.get("val_r2", np.nan)),
        "sae_train_relL2_medio": float(np.nanmean(train_rel)) if train_rel.size else np.nan,
        "sae_val_relL2_medio": float(np.nanmean(val_rel)) if val_rel.size else np.nan,
        "sae_val_relL2_p95": float(np.nanpercentile(val_rel, 95)) if val_rel.size else np.nan,
        "sae_best_epoch": scalar(z.get("best_epoch", np.nan)),
        "sae_menor_val_loss": float(np.nanmin(val_loss)) if val_loss.size else np.nan,
    }


def resumo_pod(path: Path, summary_path: Path | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    js = safe_json(summary_path) if summary_path else {}
    if js:
        out.update({
            "pod_snapshots": scalar(js.get("n_snapshots", np.nan)),
            "pod_rank": scalar(js.get("rank", np.nan)),
            "pod_energy_8": scalar(js.get("energy_at_8", np.nan)),
            "pod_energy_16": scalar(js.get("energy_at_16", np.nan)),
            "pod_energy_final": scalar(js.get("energy_at_50", np.nan)),
            "pod_error_8": scalar(js.get("error_at_8", np.nan)),
            "pod_error_16": scalar(js.get("error_at_16", np.nan)),
            "pod_error_final": scalar(js.get("error_at_50", np.nan)),
        })
        return out
    if not path.exists():
        return out
    z = np.load(path, allow_pickle=True)
    s = np.asarray(z.get("s", []), float)
    if s.size:
        e = np.cumsum(s**2) / np.sum(s**2)
        out["pod_energy_8"] = float(e[min(7, len(e)-1)])
        out["pod_energy_16"] = float(e[min(15, len(e)-1)])
        out["pod_energy_final"] = scalar(z.get("energy_at_r", e[-1]))
    out["pod_snapshots"] = scalar(z.get("Nt", len(z.get("times", []))))
    out["pod_rank"] = scalar(z.get("rank", len(s)))
    out["pod_error_final"] = scalar(z.get("err_global_rfinal", np.nan))
    return out


def resumo_sindy_summary(path: Path, prefix: str) -> dict[str, float | str]:
    js = safe_json(path)
    meta = js.get("metadata", {}) if isinstance(js.get("metadata", {}), dict) else {}
    return {
        f"{prefix}_snapshots": scalar(js.get("n_snapshots", np.nan)),
        f"{prefix}_coords": scalar(js.get("n_coordinates", np.nan)),
        f"{prefix}_dt": scalar(js.get("dt", np.nan)),
        f"{prefix}_deriv_relL2": scalar(js.get("derivative_l2_error", np.nan)),
        f"{prefix}_traj_relL2": scalar(js.get("trajectory_l2_error", np.nan)),
        f"{prefix}_traj_method": js.get("trajectory_method", ""),
        f"{prefix}_coef_nao_nulos": scalar(js.get("n_nonzero_terms", np.nan)),
        f"{prefix}_library_size": scalar(js.get("library_size", np.nan)),
        f"{prefix}_pod_energy_capture": scalar(meta.get("pod_energy_capture", np.nan)),
        f"{prefix}_integration_message": js.get("integration_message", ""),
    }


def resumo_hibrido(folder: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, str]]]:
    z = np.load(folder / "best_model.npz", allow_pickle=True)
    states = np.asarray(z["states"], float)
    free = np.asarray(z["free_pred"], float)
    one = np.asarray(z["one_step_pred"], float)
    dref = np.asarray(z["derivatives_reference"], float)
    dpred = np.asarray(z["derivatives_pred"], float)
    times = np.asarray(z["times"], float)
    Xi = np.asarray(z["Xi"], float)
    rows = read_csv_dicts(folder / "sindy_search_results.csv")
    best_csv = min(rows, key=lambda rr: frow(rr, "free_rel_l2", np.inf)) if rows else {}
    summary = safe_json(folder / "summary.json")
    err_free_t = rel_l2(states, free, axis=1)
    err_one_t = rel_l2(states, one, axis=1)
    err_deriv_t = rel_l2(dref, dpred, axis=1)
    peak = int(np.nanargmax(err_free_t))
    info = {
        "hyb_snapshots": len(times),
        "hyb_dt_medio": float(np.nanmean(np.diff(times))) if len(times) > 1 else np.nan,
        "hyb_modulo_temporal_s": float(times[-1] - times[0]) if len(times) > 1 else np.nan,
        "hyb_latent_pod_modes": frow(best_csv, "latent_pod_modes"),
        "hyb_latent_pod_energy": frow(best_csv, "latent_pod_energy_capture"),
        "hyb_alpha": frow(best_csv, "alpha"),
        "hyb_smooth_window": frow(best_csv, "smooth_window"),
        "hyb_library_size": frow(best_csv, "library_size"),
        "hyb_n_possible_terms": int(Xi.size),
        "hyb_n_nonzero_terms": int(np.count_nonzero(np.abs(Xi) > 1e-12)),
        "hyb_sparsity": 1 - int(np.count_nonzero(np.abs(Xi) > 1e-12)) / max(int(Xi.size), 1),
        "hyb_deriv_relL2": float(rel_l2(dref, dpred)),
        "hyb_one_step_relL2": float(rel_l2(states, one)),
        "hyb_free_relL2": float(rel_l2(states, free)),
        "hyb_free_r2": r2(states, free),
        "hyb_deriv_r2": r2(dref, dpred),
        "hyb_aic": frow(best_csv, "aic"),
        "hyb_n_tested": int(summary.get("n_tested", len(rows))),
        "hyb_n_accepted": int(summary.get("n_accepted", sum(str(rr.get("accepted", "")).lower() == "true" for rr in rows))),
        "hyb_peak_error": float(err_free_t[peak]),
        "hyb_peak_time": float(times[peak]),
        "hyb_final_error": float(err_free_t[-1]),
        "hyb_p50_error": float(np.nanpercentile(err_free_t, 50)),
        "hyb_p95_error": float(np.nanpercentile(err_free_t, 95)),
    }
    temporal = {"times": times, "free": err_free_t, "one": err_one_t, "deriv": err_deriv_t}
    return info, temporal, rows


def carregar_tudo():
    linhas: list[dict[str, Any]] = []
    temporal: dict[str, dict[str, np.ndarray]] = {}
    busca_rows: list[dict[str, Any]] = []
    for caso, cfg in CASOS.items():
        hyb, temp, rows = resumo_hibrido(cfg["hibrido"])
        row: dict[str, Any] = {"caso": caso}
        row.update(resumo_pod(cfg["pod"], cfg.get("pod_summary")))
        row.update(resumo_sae(cfg["sae"]))
        row.update(resumo_sindy_summary(cfg["pod_sindy"], "pod_sindy"))
        row.update(resumo_sindy_summary(cfg["sae_sindy"], "sae_sindy"))
        row.update(hyb)
        linhas.append(row)
        temporal[caso] = temp
        for rr in rows:
            out = {"caso": caso}
            for k, v in rr.items():
                out[k] = v
            busca_rows.append(out)
    return linhas, temporal, busca_rows


def escrever_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def gerar_tabelas(linhas, temporal, busca_rows):
    escrever_csv(OUT / "comparacao_resultados_completa.csv", linhas)
    escrever_csv(OUT / "comparacao_busca_modelos_sindy.csv", busca_rows)
    resumo_temp = []
    for caso, t in temporal.items():
        for nome in ["free", "one", "deriv"]:
            y = t[nome]
            resumo_temp.append({
                "caso": caso, "tipo_erro": nome,
                "min": float(np.nanmin(y)), "p25": float(np.nanpercentile(y, 25)),
                "mediana": float(np.nanpercentile(y, 50)), "p75": float(np.nanpercentile(y, 75)),
                "p95": float(np.nanpercentile(y, 95)), "max": float(np.nanmax(y)),
                "tempo_max": float(t["times"][int(np.nanargmax(y))]),
            })
    escrever_csv(OUT / "comparacao_erro_temporal_resumo.csv", resumo_temp)


def gerar_figuras(linhas, temporal, busca_rows):
    labels = [r["caso"] for r in linhas]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5))
    width = .22
    ax.bar(x - width, [float(r["pod_sindy_deriv_relL2"]) for r in linhas], width, label="POD--SINDy deriv.")
    ax.bar(x, [float(r["sae_sindy_deriv_relL2"]) for r in linhas], width, label="SAE--SINDy deriv.")
    ax.bar(x + width, [float(r["hyb_deriv_relL2"]) for r in linhas], width, label="SAE--POD--SINDy deriv.")
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("erro relativo $L_2$ das derivadas")
    ax.set_title("Comparação de ajuste local entre arquiteturas")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_comparacao_derivadas_arquiteturas.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, [float(r["hyb_deriv_relL2"]) for r in linhas], width, label="derivadas")
    ax.bar(x, [float(r["hyb_one_step_relL2"]) for r in linhas], width, label="um passo")
    ax.bar(x + width, [float(r["hyb_free_relL2"]) for r in linhas], width, label="integração livre")
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("erro relativo $L_2$")
    ax.set_title("SAE--POD--SINDy: erro local versus erro acumulado")
    ax.grid(axis="y", alpha=.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_hibrido_erros_globais.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(labels, [float(r["pod_energy_8"]) for r in linhas], "o-", label="POD dados originais: energia 8 modos")
    ax.plot(labels, [float(r["hyb_latent_pod_energy"]) for r in linhas], "s-", label="POD no latente SAE: energia usada")
    ax2 = ax.twinx()
    ax2.plot(labels, [float(r["hyb_free_relL2"]) for r in linhas], "^-", color="#d62728", label="erro livre híbrido")
    ax.set_ylim(0, 1.05); ax.set_ylabel("energia acumulada")
    ax2.set_ylabel("erro livre relativo")
    ax.set_title("Energia capturada versus erro de previsão livre")
    ax.grid(axis="y", alpha=.25)
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=9, loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "fig_energia_vs_erro_livre.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    for ax, caso in zip(axes.ravel(), labels):
        t = temporal[caso]["times"] - temporal[caso]["times"][0]
        ax.plot(t, temporal[caso]["free"], lw=2, label="livre")
        ax.plot(t, temporal[caso]["one"], lw=1.2, label="um passo")
        ax.plot(t, temporal[caso]["deriv"], lw=1.0, label="derivadas")
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_title(caso); ax.set_xlabel("tempo (s)"); ax.set_ylabel("erro inst.")
        ax.grid(True, alpha=.25)
    axes.ravel()[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_erros_temporais_por_caso.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter([float(r["hyb_sparsity"]) for r in linhas], [float(r["hyb_free_relL2"]) for r in linhas], s=90)
    for r in linhas:
        ax.annotate(r["caso"], (float(r["hyb_sparsity"]), float(r["hyb_free_relL2"])), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("esparsidade do modelo híbrido")
    ax.set_ylabel("erro livre relativo")
    ax.set_title("Parcimônia versus fidelidade da integração livre")
    ax.grid(True, alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_esparsidade_vs_erro.png", dpi=300); plt.close(fig)


def carregar_periodos_temporais() -> list[dict[str, Any]]:
    """Lê temporal_error_by_period.csv de cada caso híbrido."""
    rows: list[dict[str, Any]] = []
    for caso, cfg in CASOS.items():
        path = cfg["hibrido"] / "temporal_error_by_period.csv"
        for r in read_csv_dicts(path):
            out = {"caso": caso}
            out.update(r)
            rows.append(out)
    return rows


def carregar_relatorios_campos() -> list[dict[str, Any]]:
    """Lê report.json com erros por campo nos tempos alvo."""
    rows: list[dict[str, Any]] = []
    for caso, cfg in CASOS.items():
        path = cfg["hibrido"] / "field_comparisons_exact_times" / "report.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        for item in data:
            base = {
                "caso": caso,
                "requested_time": item.get("requested_time", item.get("target_time", np.nan)),
                "actual_time": item.get("actual_time", item.get("sindy_time", np.nan)),
                "index": item.get("index", item.get("sindy_index", np.nan)),
                "trajectory": item.get("trajectory", ""),
            }
            campos = sorted({k[:-7] for k in item if k.endswith("_rel_l2")})
            for campo in campos:
                rows.append({
                    **base,
                    "campo": campo,
                    "rel_l2": item.get(f"{campo}_rel_l2", np.nan),
                    "r2": item.get(f"{campo}_r2", np.nan),
                })
    return rows


def carregar_erros_sae_validacao() -> list[dict[str, Any]]:
    """Extrai a distribuição de erros SAE de validação para boxplots e tabelas."""
    rows: list[dict[str, Any]] = []
    for caso, cfg in CASOS.items():
        path = cfg["sae"]
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        for i, val in enumerate(np.asarray(z.get("val_relL2", []), float)):
            rows.append({"caso": caso, "indice_validacao": i, "val_relL2": float(val)})
    return rows


def resumir_busca_parametros(busca_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Agrega a busca SINDy por alpha, janela de suavização e integrador."""
    def agg(keys: list[str]) -> list[dict[str, Any]]:
        grupos: dict[tuple, list[dict[str, Any]]] = {}
        for r in busca_rows:
            grupos.setdefault(tuple(r.get(k, "") for k in keys), []).append(r)
        out = []
        for vals, rs in grupos.items():
            free = np.array([frow(r, "free_rel_l2") for r in rs], float)
            deriv = np.array([frow(r, "derivative_rel_l2") for r in rs], float)
            acc = np.array([str(r.get("accepted", "")).lower() == "true" for r in rs], bool)
            row = {k: v for k, v in zip(keys, vals)}
            row.update({
                "n": len(rs),
                "aceitos": int(acc.sum()),
                "taxa_aceitacao": float(acc.mean()) if len(acc) else np.nan,
                "free_rel_l2_min": float(np.nanmin(free)) if free.size else np.nan,
                "free_rel_l2_mediana": float(np.nanmedian(free)) if free.size else np.nan,
                "derivative_rel_l2_mediana": float(np.nanmedian(deriv)) if deriv.size else np.nan,
            })
            out.append(row)
        return out

    por_alpha = agg(["caso", "alpha"])
    por_smooth = agg(["caso", "smooth_window"])
    por_integrador = agg(["caso", "integrator"])

    correl = []
    for caso in sorted({r["caso"] for r in busca_rows}):
        rs = [r for r in busca_rows if r["caso"] == caso]
        x = np.array([frow(r, "derivative_rel_l2") for r in rs], float)
        y = np.array([frow(r, "free_rel_l2") for r in rs], float)
        m = np.isfinite(x) & np.isfinite(y) & (y > 0)
        corr = float(np.corrcoef(x[m], np.log10(y[m]))[0, 1]) if m.sum() > 2 else np.nan
        correl.append({"caso": caso, "n": int(m.sum()), "corr_deriv_vs_log_free": corr})
    return por_alpha, por_smooth, por_integrador, correl


def gerar_tabelas_extras(busca_rows):
    periodos = carregar_periodos_temporais()
    campos = carregar_relatorios_campos()
    sae_val = carregar_erros_sae_validacao()
    por_alpha, por_smooth, por_integrador, correl = resumir_busca_parametros(busca_rows)
    escrever_csv(OUT / "comparacao_parametros_alpha.csv", por_alpha)
    escrever_csv(OUT / "comparacao_parametros_smooth_window.csv", por_smooth)
    escrever_csv(OUT / "comparacao_parametros_integrador.csv", por_integrador)
    escrever_csv(OUT / "comparacao_correlacao_erro_local_livre.csv", correl)
    escrever_csv(OUT / "comparacao_periodos_temporais.csv", periodos)
    escrever_csv(OUT / "comparacao_erros_por_campo.csv", campos)
    escrever_csv(OUT / "comparacao_sae_erros_validacao.csv", sae_val)
    return periodos, campos, sae_val, por_alpha, por_smooth, por_integrador, correl


def gerar_figuras_imagens_existentes():
    """Monta painéis com imagens já geradas pelos casos SAE--POD--SINDy."""
    itens_heatmap = [(caso, cfg["hibrido"] / "temporal_learning_heatmap.png") for caso, cfg in CASOS.items()]
    itens_curvas_erro = [(caso, cfg["hibrido"] / "temporal_error_curves.png") for caso, cfg in CASOS.items()]
    itens_campos = [
        ("Laminar t=0,50 s", CASOS["Laminar"]["hibrido"] / "field_comparisons_exact_times" / "time_0.50s.png"),
        ("URANS t=0,50 s", CASOS["URANS"]["hibrido"] / "field_comparisons_exact_times" / "time_0.50s.png"),
        ("LES--WALE t=0,50 s", CASOS["LES--WALE"]["hibrido"] / "field_comparisons_exact_times" / "time_0.50s.png"),
        ("Cavity t=0,75 s", CASOS["Cavity"]["hibrido"] / "field_comparisons_exact_times" / "time_0.75s.png"),
    ]

    def painel(itens, saida, titulo):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, (rotulo, path) in zip(axes.ravel(), itens):
            if path.exists():
                img = plt.imread(path)
                ax.imshow(img)
                ax.set_title(rotulo, fontsize=11)
            else:
                ax.text(0.5, 0.5, f"Imagem não encontrada\n{path}", ha="center", va="center", fontsize=8)
                ax.set_title(rotulo, fontsize=11)
            ax.axis("off")
        fig.suptitle(titulo, fontsize=14)
        fig.tight_layout()
        fig.savefig(OUT / saida, dpi=300, bbox_inches="tight")
        plt.close(fig)

    painel(
        itens_heatmap,
        "fig_painel_temporal_learning_heatmap.png",
        "Mapas de aprendizado temporal dos modelos SAE--POD--SINDy",
    )
    painel(
        itens_curvas_erro,
        "fig_painel_temporal_error_curves.png",
        "Curvas temporais de erro dos modelos SAE--POD--SINDy",
    )
    painel(
        itens_campos,
        "fig_painel_field_comparisons_tempos_alvo.png",
        "Comparações de campos SAE--POD--SINDy em tempos alvo",
    )


def gerar_figuras_extras(busca_rows, periodos, campos, sae_val, por_alpha, por_smooth, por_integrador, correl):
    labels = list(CASOS.keys())
    colors = {c: plt.cm.tab10(i) for i, c in enumerate(labels)}

    # Alpha x erro livre.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for caso in labels:
        rs = [r for r in busca_rows if r["caso"] == caso]
        x = np.array([frow(r, "alpha") for r in rs], float)
        y = np.array([frow(r, "free_rel_l2") for r in rs], float)
        acc = np.array([str(r.get("accepted", "")).lower() == "true" for r in rs], bool)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        ax.scatter(x[m & ~acc], y[m & ~acc], s=24, color=colors[caso], alpha=.25, marker="x")
        ax.scatter(x[m & acc], y[m & acc], s=44, color=colors[caso], alpha=.85, label=caso)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("alpha da regressão")
    ax.set_ylabel("erro livre relativo")
    ax.set_title("Sensibilidade do SINDy ao alpha: pontos cheios são aceitos")
    ax.grid(True, alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_alpha_vs_erro_livre.png", dpi=300); plt.close(fig)

    # Smooth window x erro livre.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for caso in labels:
        rs = [r for r in busca_rows if r["caso"] == caso]
        x = np.array([frow(r, "smooth_window") for r in rs], float)
        y = np.array([frow(r, "free_rel_l2") for r in rs], float)
        acc = np.array([str(r.get("accepted", "")).lower() == "true" for r in rs], bool)
        m = np.isfinite(x) & np.isfinite(y) & (y > 0)
        ax.scatter(x[m & ~acc], y[m & ~acc], s=24, color=colors[caso], alpha=.25, marker="x")
        ax.scatter(x[m & acc], y[m & acc], s=44, color=colors[caso], alpha=.85, label=caso)
    ax.set_yscale("log")
    ax.set_xlabel("janela de suavização")
    ax.set_ylabel("erro livre relativo")
    ax.set_title("Sensibilidade à suavização das derivadas")
    ax.grid(True, alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_smooth_window_vs_erro_livre.png", dpi=300); plt.close(fig)

    # Taxa de aceitação por integrador.
    integradores = sorted({r.get("integrator", "") for r in por_integrador})
    x = np.arange(len(labels)); width = 0.8 / max(len(integradores), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, integ in enumerate(integradores):
        vals = []
        for caso in labels:
            match = [r for r in por_integrador if r.get("caso") == caso and r.get("integrator") == integ]
            vals.append(float(match[0]["taxa_aceitacao"]) if match else np.nan)
        ax.bar(x + (j - (len(integradores)-1)/2)*width, vals, width, label=integ)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("taxa de aceitação")
    ax.set_title("Sucesso da integração livre por integrador")
    ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_integrador_taxa_aceitacao.png", dpi=300); plt.close(fig)

    # Erro por período temporal.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for caso in labels:
        rs = [r for r in periodos if r["caso"] == caso]
        rs = sorted(rs, key=lambda r: int(float(r.get("period", 0))))
        x = [int(float(r["period"])) for r in rs]
        y = [frow(r, "rel_l2") for r in rs]
        ax.plot(x, y, marker="o", lw=1.8, label=caso)
    ax.set_yscale("log")
    ax.set_xlabel("período temporal da janela")
    ax.set_ylabel("erro relativo $L_2$ por período")
    ax.set_title("Onde o erro livre cresce ao longo da simulação")
    ax.grid(True, alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_erro_por_periodo_temporal.png", dpi=300); plt.close(fig)

    # Erro por campo: média nos tempos alvo.
    campos_unicos = sorted({r["campo"] for r in campos})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels)); width = 0.8 / max(len(campos_unicos), 1)
    for j, campo in enumerate(campos_unicos):
        vals = []
        for caso in labels:
            ys = [float(r["rel_l2"]) for r in campos if r["caso"] == caso and r["campo"] == campo and np.isfinite(float(r["rel_l2"]))]
            vals.append(float(np.nanmedian(ys)) if ys else np.nan)
        ax.bar(x + (j - (len(campos_unicos)-1)/2)*width, vals, width, label=campo)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("mediana do erro relativo por campo")
    ax.set_title("Validação por campo nos tempos alvo")
    ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_erro_por_campo.png", dpi=300); plt.close(fig)

    # Boxplot SAE.
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [[float(r["val_relL2"]) for r in sae_val if r["caso"] == caso] for caso in labels]
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("erro relativo $L_2$ de validação do SAE")
    ax.set_title("Distribuição dos erros de reconstrução SAE")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_sae_boxplot_val_relL2.png", dpi=300); plt.close(fig)

    # Correlação erro local x erro livre.
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [float(next((r["corr_deriv_vs_log_free"] for r in correl if r["caso"] == caso), np.nan)) for caso in labels]
    ax.bar(labels, vals, color=[colors[c] for c in labels])
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("corr(erro derivadas, log10 erro livre)")
    ax.set_title("Quanto o erro local explica o erro acumulado?")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_correlacao_erro_local_livre.png", dpi=300); plt.close(fig)


def main():
    linhas, temporal, busca_rows = carregar_tudo()
    gerar_tabelas(linhas, temporal, busca_rows)
    gerar_figuras(linhas, temporal, busca_rows)
    extras = gerar_tabelas_extras(busca_rows)
    gerar_figuras_extras(busca_rows, *extras)
    gerar_figuras_imagens_existentes()
    print("Arquivos gerados em:", OUT)
    for p in sorted(OUT.glob("comparacao_*.csv")):
        print(" -", p.name)
    for p in sorted(OUT.glob("fig_*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
