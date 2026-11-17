"""
scripts/analise_lei_escala_pop_media_raio_fixo.py
-------------------------------------------------
Implementa a Etapa 2 das diretrizes do Prof. Dr. Erneson Oliveira (Audios 1, 2 e 4):
  1. Carrega 100% das bacias costeiras extraidas via `externos/GebcoCutter` + `externos/IPBA`
     (corte h = 0, sem filtro de area) e populacao via `externos/CCA` (GPW v4),
     consolidadas em `relatorios/bacias_globais_gebco_cutter_ipba_cca.csv`.
  2. Para cada raio fixo R (mesmo tamanho de circulo para todas as fozes/sinks), calcula a
     MEDIA aritmetica (pieces/m3, nunca a soma!) dos pontos de coleta dentro do circulo.
  3. Responde a pergunta de otimizacao do raio R:
     "Para raio R1 estimamos para N1 bacias com R² e beta; para R2 estimamos para N2 bacias..."
     comparando:
       (i) Todas as bacias (100%, incluindo microbacias costeiras - "sujeira" diagnostica do Audio 2);
       (ii) Maiores bacias do IPBA (Top 95% da area continental);
       (iii) Maiores bacias com circulos exclusivos (`shared_with_other_sink == False`, Audio 4).
  4. Gera 3 figuras cientificas em PNG (300 DPI) e PDF vetorial em `relatorios/figuras/`:
       - `noaa_lei_escala_todas_vs_maiores_vs_exclusivas.(png|pdf)`
       - `noaa_comparacao_raios_fixos_lei_escala.(png|pdf)`
       - `noaa_scatter_por_tipo_lixo_media_raio.(png|pdf)`
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from geracao_graficos.estilo_erneson import aplicar_estilo, rotulo_painel, salvar
from pipeline.ipba_runner import EARTH_RADIUS_KM
from scripts.analise_r22km_mar_territorial import (
    carregar_ocean_water_filtrado_mar,
    carregar_ou_gerar_bacias_globais_externos,
)


def to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def fmt_int_pt(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_dec_pt(val: float, dec: int = 2) -> str:
    return f"{val:.{dec}f}".replace(".", ",")


def extrair_ou_carregar_bacias_100pct(ow_sea: pd.DataFrame) -> pd.DataFrame:
    """
    Carrega o catalogo global de bacias gerado por `externos/GebcoCutter` + `externos/IPBA`
    + `externos/CCA` (`relatorios/bacias_globais_gebco_cutter_ipba_cca.csv`).
    """
    return carregar_ou_gerar_bacias_globais_externos(ow_sea)


def carregar_base_noaa() -> pd.DataFrame:
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    with open(noaa_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    df = pd.DataFrame([ft["properties"] for ft in gj["features"]])
    df["lat"] = pd.to_numeric(df["Latitude__degree_"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude_degree_"], errors="coerce")
    df["mesh_mm"] = pd.to_numeric(df["Mesh_size__mm_"], errors="coerce")
    df["mp_val"] = pd.to_numeric(df["Microplastics_measurement"], errors="coerce")
    df["nurdle_val"] = pd.to_numeric(df["Standardized_Nurdle__Amount"], errors="coerce")
    df["valor"] = np.where(df["Medium"] == "Nurdle Patrol", df["nurdle_val"], df["mp_val"])
    return df[df["lat"].notna() & df["lon"].notna() & df["valor"].notna()].reset_index(drop=True)


def calcular_media_raio_fixo_nos_sinks(
    sinks_df: pd.DataFrame,
    pts_df: pd.DataFrame,
    r_km: float,
) -> pd.DataFrame:
    """
    Associa a cada sink um circulo de raio fixo `r_km` e calcula a MEDIA aritmetica
    dos valores dos pontos que caem dentro do circulo, alem da flag `shared_with_other_sink`.
    """
    out = sinks_df.copy().reset_index(drop=True)
    s_xyz = to_xyz(out["latitude"].to_numpy(), out["longitude"].to_numpy())
    p_xyz = to_xyz(pts_df["lat"].to_numpy(), pts_df["lon"].to_numpy())

    tree_p = cKDTree(p_xyz)
    tree_s = cKDTree(s_xyz)

    chord = float(2.0 * np.sin((r_km / EARTH_RADIUS_KM) / 2.0))
    p_lists = tree_p.query_ball_point(s_xyz, r=chord)
    s_lists = tree_s.query_ball_point(p_xyz, r=chord)
    pt_usage = np.array([len(x) for x in s_lists], dtype=int)

    vals = pts_df["valor"].to_numpy()
    means = np.full(len(out), np.nan, dtype=float)
    stds = np.full(len(out), np.nan, dtype=float)
    n_pts = np.zeros(len(out), dtype=int)
    shared = np.zeros(len(out), dtype=bool)
    max_use = np.zeros(len(out), dtype=int)

    for i, idx_list in enumerate(p_lists):
        if len(idx_list) > 0:
            arr_idx = np.array(idx_list, dtype=int)
            v = vals[arr_idx]
            u = pt_usage[arr_idx]
            means[i] = float(np.mean(v))
            stds[i] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            n_pts[i] = len(v)
            shared[i] = bool(np.any(u > 1))
            max_use[i] = int(np.max(u))

    out["mp_mean"] = means
    out["mp_std"] = stds
    out["n_points"] = n_pts
    out["flux"] = out["mp_mean"] * out["area_km2"]
    out["shared_with_other_sink"] = shared
    out["max_point_usage"] = max_use
    return out


def _ajustar_ols(df_sub: pd.DataFrame, pop_col: str, y_col: str = "mp_mean") -> dict[str, Any]:
    with_pts = df_sub[(df_sub[pop_col] > 0) & (df_sub["n_points"] > 0)]
    valid = df_sub[(df_sub[pop_col] > 0) & (df_sub[y_col] > 0) & np.isfinite(df_sub[y_col])].copy()
    n = len(valid)
    n_com_pts = len(with_pts)
    if n < 4:
        return {
            "N_com_pts": int(n_com_pts),
            "N": int(n),
            "beta": np.nan,
            "se": np.nan,
            "r": np.nan,
            "r2": np.nan,
            "p_val": np.nan,
            "rho": np.nan,
            "p_rho": np.nan,
            "r2_bins": np.nan,
            "p_bins": np.nan,
        }
    x = np.log10(valid[pop_col].to_numpy())
    y = np.log10(valid[y_col].to_numpy())
    sl, ic, rv, pv, se = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)

    r2_bins = np.nan
    p_bins = np.nan
    if n >= 20:
        bins = np.linspace(x.min(), x.max(), 9)
        valid["bin"] = pd.cut(x, bins=bins, include_lowest=True)
        b_df = (
            valid.groupby("bin", observed=True)
            .agg(
                n_bin=(y_col, "count"),
                xm=(pop_col, lambda s: float(np.mean(np.log10(s)))),
                ym=(y_col, lambda s: float(np.mean(np.log10(s)))),
            )
            .dropna()
        )
        b_df = b_df[b_df["n_bin"] >= 3]
        if len(b_df) >= 4:
            _, _, rv_b, pv_b, _ = stats.linregress(b_df["xm"].to_numpy(), b_df["ym"].to_numpy())
            r2_bins = float(rv_b**2)
            p_bins = float(pv_b)

    return {
        "N_com_pts": int(n_com_pts),
        "N": int(n),
        "beta": float(sl),
        "intercept": float(ic),
        "se": float(se),
        "r": float(rv),
        "r2": float(rv**2),
        "p_val": float(pv),
        "rho": float(rho),
        "p_rho": float(prho),
        "r2_bins": r2_bins,
        "p_bins": p_bins,
    }


def avaliar_sensibilidade_raios_lei_escala(
    df_bacias_100: pd.DataFrame,
    ow_df: pd.DataFrame,
    raios: list[int] | None = None,
) -> pd.DataFrame:
    """
    Testa todos os raios inteiros de 1 a 22 km (passo 1 km), fixando o mesmo R para todas as fozes,
    e registra para cada R quantas bacias sao estimadas (N) e a qualidade da Lei de Potencia
    (Concentracao Media e Fluxo = Media * Area) nos 4 regimes:
      1. Todas as bacias (100% dos sinks h = 0, sem corte de area)
      2. Bacias principais (Area >= 100 km², todas)
      3. Bacias principais com circulos exclusivos (`shared_with_other_sink == False`)
      4. Top 95% global de area acumulada
    """
    if raios is None:
        raios = list(range(1, 23))

    df_a100_base = df_bacias_100[df_bacias_100["area_km2"] >= 100.0].copy().reset_index(drop=True)
    df_95g_base = df_bacias_100[df_bacias_100["in_top_95pct_global"]].copy().reset_index(drop=True)

    rows = []
    for r_km in raios:
        m_100 = calcular_media_raio_fixo_nos_sinks(df_bacias_100, ow_df, r_km)
        m_a100 = calcular_media_raio_fixo_nos_sinks(df_a100_base, ow_df, r_km)
        m_a100_excl = m_a100[~m_a100["shared_with_other_sink"]].copy()
        m_95g = calcular_media_raio_fixo_nos_sinks(df_95g_base, ow_df, r_km)

        for regime_nome, df_reg in [
            ("1_Todas_Bacias_100pct_Sem_Corte", m_100),
            ("2_Bacias_Area_ge_100km2_Todas", m_a100),
            ("3_Bacias_Area_ge_100km2_Exclusivas", m_a100_excl),
            ("4_Maiores_Bacias_Top95pct_Global", m_95g),
        ]:
            for pop_col in ["urban_population", "total_population"]:
                for metric_nome, y_col in [("concentracao", "mp_mean"), ("fluxo", "flux")]:
                    res = _ajustar_ols(df_reg, pop_col, y_col)
                    r2_pct = res["r2"] * 100.0 if np.isfinite(res["r2"]) else np.nan
                    r2_bins_pct = res["r2_bins"] * 100.0 if np.isfinite(res["r2_bins"]) else np.nan
                    rows.append(
                        {
                            "r_km": int(r_km),
                            "regime": regime_nome,
                            "pop_col": pop_col,
                            "metrica": metric_nome,
                            "n_bacias_com_pts_mar": res["N_com_pts"],
                            "n_bacias_estimadas": res["N"],
                            "beta": res["beta"],
                            "se": res["se"],
                            "r": res["r"],
                            "r2_pct": r2_pct,
                            "p_value": res["p_val"],
                            "spearman_rho": res["rho"],
                            "spearman_p": res["p_rho"],
                            "r2_bins_pct": r2_bins_pct,
                            "p_bins": res["p_bins"],
                            "score_sqrtN_R2": float(np.sqrt(res["N"]) * r2_pct) if np.isfinite(r2_pct) else np.nan,
                            "produto_N_R2": float(res["N"] * r2_pct) if np.isfinite(r2_pct) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def selecionar_melhores_raios_1_a_22km(df_sens: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """
    Identifica automaticamente qual raio R em [1..22 km] atinge o melhor resultado segundo
    criterios objetivos (maximo R² significativo, melhor equilibrio cobertura x R², e maxima cobertura).
    """
    sub = df_sens[
        (df_sens["r_km"].between(1, 22))
        & (df_sens["n_bacias_estimadas"] >= min_n)
        & (df_sens["beta"] > 0)
        & (df_sens["p_value"] < 0.001)
    ].copy()

    criterios = [
        ("Max R² — Urb. CCA Concentracao (Area >= 100 km²)", "2_Bacias_Area_ge_100km2_Todas", "urban_population", "concentracao", "r2_pct", None),
        ("Max R² — Urb. CCA Fluxo (Area >= 100 km²)", "2_Bacias_Area_ge_100km2_Todas", "urban_population", "fluxo", "r2_pct", None),
        ("Max R² — Circulos Exclusivos Conc. (Area >= 100 km²)", "3_Bacias_Area_ge_100km2_Exclusivas", "urban_population", "concentracao", "r2_pct", None),
        ("Max R² — Pop. Total Concentracao (Area >= 100 km²)", "2_Bacias_Area_ge_100km2_Todas", "total_population", "concentracao", "r2_pct", None),
        ("Max R² — Pop. Total Fluxo (Area >= 100 km²)", "2_Bacias_Area_ge_100km2_Todas", "total_population", "fluxo", "r2_pct", None),
        ("Compromisso Intermediario (N >= 150 Urb / N ~ 300 Tot) — Max R²", "2_Bacias_Area_ge_100km2_Todas", "urban_population", "concentracao", "r2_pct", lambda d: d["n_bacias_estimadas"] >= 150),
        ("Max N * R² — Urb. CCA Concentracao", "2_Bacias_Area_ge_100km2_Todas", "urban_population", "concentracao", "produto_N_R2", None),
        ("Max sqrt(N) * R² — Pop. Total Fluxo (Cobertura x Ajuste)", "2_Bacias_Area_ge_100km2_Todas", "total_population", "fluxo", "score_sqrtN_R2", None),
    ]

    vencedores = []
    for nome_c, reg, p_col, met, obj_col, extra_fn in criterios:
        cand = sub[(sub["regime"] == reg) & (sub["pop_col"] == p_col) & (sub["metrica"] == met)]
        if extra_fn is not None:
            cand = cand[extra_fn(cand)]
        if cand.empty or cand[obj_col].isna().all():
            continue
        best = cand.loc[cand[obj_col].idxmax()]
        vencedores.append(
            {
                "criterio": nome_c,
                "r_otimo_km": int(best["r_km"]),
                "n_bacias_estimadas": int(best["n_bacias_estimadas"]),
                "beta": float(best["beta"]),
                "se": float(best["se"]),
                "r": float(best["r"]),
                "r2_pct": float(best["r2_pct"]),
                "r2_bins_pct": float(best["r2_bins_pct"]) if np.isfinite(best["r2_bins_pct"]) else np.nan,
                "p_value": float(best["p_value"]),
            }
        )
    return pd.DataFrame(vencedores)


def _plotar_scatter_alometria(
    ax: plt.Axes,
    df_sub: pd.DataFrame,
    pop_col: str,
    titulo: str,
    cor_pts: str,
    cor_reta: str,
    xlabel: str,
    ylabel: str,
    destacar_microbacias: bool = False,
) -> None:
    valid = df_sub[(df_sub[pop_col] > 0) & (df_sub["mp_mean"] > 0) & np.isfinite(df_sub["mp_mean"])].copy()
    if len(valid) < 4:
        ax.text(0.5, 0.5, "Dados insuficientes", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.log10(valid[pop_col].to_numpy())
    y = np.log10(valid["mp_mean"].to_numpy())

    if destacar_microbacias and "area_km2" in valid.columns:
        m_micro = (valid["area_km2"] < 100.0).to_numpy()
        m_maj = (valid["area_km2"] >= 100.0).to_numpy()
        if np.any(m_micro):
            ax.scatter(
                x[m_micro],
                y[m_micro],
                edgecolors="#f59e0b",
                facecolors="none",
                marker="o",
                s=26,
                linewidths=0.9,
                alpha=0.72,
                label=f"Microbacias < 100 km² (n = {int(m_micro.sum())})",
                zorder=3,
            )
        if np.any(m_maj):
            ax.scatter(
                x[m_maj],
                y[m_maj],
                edgecolors=cor_pts,
                facecolors="none",
                marker="o",
                s=34,
                linewidths=1.1,
                alpha=0.88,
                label=f"Bacias ≥ 100 km² (n = {int(m_maj.sum())})",
                zorder=4,
            )
    else:
        ax.scatter(
            x,
            y,
            edgecolors=cor_pts,
            facecolors="none",
            marker="o",
            s=38,
            linewidths=1.2,
            alpha=0.90,
            label=f"Bacias estimadas (N = {len(valid)})",
            zorder=4,
        )

    sl, ic, rv, pv, se = stats.linregress(x, y)
    rho, _ = stats.spearmanr(x, y)

    xl = np.linspace(x.min() - 0.12, x.max() + 0.12, 100)
    yl = ic + sl * xl
    ax.plot(xl, yl, color=cor_reta, linewidth=1.8, zorder=6, label="Ajuste OLS (Lei de Potência)")

    # Banda IC 95% da regressao
    n = len(x)
    if n > 3:
        t_val = float(stats.t.ppf(0.975, df=n - 2))
        s_err = float(np.sqrt(np.sum((y - (ic + sl * x)) ** 2) / (n - 2)))
        ci = t_val * s_err * np.sqrt(1.0 / n + (xl - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        ax.fill_between(xl, yl - ci, yl + ci, color=cor_reta, alpha=0.15, zorder=2)

    # Referencia isometrica beta = 1
    ax.plot(
        xl,
        np.mean(y) + 1.0 * (xl - np.mean(x)),
        color="#64748b",
        linestyle=":",
        linewidth=1.1,
        zorder=3,
        label=r"Referência linear ($\beta = 1{,}0$)",
    )

    def _ltx_num(v: float, dec: int = 2, signed: bool = False) -> str:
        s = f"{v:+.{dec}f}" if signed else f"{v:.{dec}f}"
        return s.replace(".", "{,}")

    p_str = "p < 0{,}001" if pv < 0.001 else f"p = {_ltx_num(pv, 4, False)}"
    texto_stat = (
        f"$\\beta = {_ltx_num(sl, 2, True)} \\pm {_ltx_num(se, 2, False)}$\n"
        f"$R^2 = {_ltx_num(rv**2 * 100, 1, False)}\\%$ ($r = {_ltx_num(rv, 2, True)}$)\n"
        f"Spearman $\\rho = {_ltx_num(rho, 2, True)}$ (${p_str}$)\n"
        f"$N = {n}$ bacias"
    )

    ax.text(
        0.04,
        0.96,
        texto_stat,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cccccc", alpha=0.94),
        zorder=10,
    )

    ax.set_title(titulo, fontsize=10.3, pad=9)
    ax.set_xlabel(xlabel, fontsize=9.8)
    ax.set_ylabel(ylabel, fontsize=9.8)
    y_span = max(y.max() - y.min(), 3.0)
    ax.set_ylim(y.min() - 0.34 * y_span, y.max() + 0.44 * y_span)
    ax.legend(
        loc="lower right",
        fontsize=7.3,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.94,
    )


def gerar_figura_todas_vs_maiores_vs_exclusivas(
    df_bacias_100: pd.DataFrame,
    ow_df: pd.DataFrame,
    r_km: float = 5.0,
) -> Path:
    """
    Gera a figura comparativa dos 3 estagios de filtragem para o raio otimo (R = 5 km):
      - Painel (a): 100% das bacias h=0 (incluindo microbacias < 100 km²).
      - Painel (b): Bacias principais (Area >= 100 km²), permitindo sobreposicao de circulos.
      - Painel (c): Bacias principais (Area >= 100 km²) com circulos exclusivos (`shared == False`).
    """
    aplicar_estilo()

    m_100 = calcular_media_raio_fixo_nos_sinks(df_bacias_100, ow_df, r_km)
    df_a100 = df_bacias_100[df_bacias_100["area_km2"] >= 100.0].copy().reset_index(drop=True)
    m_a100 = calcular_media_raio_fixo_nos_sinks(df_a100, ow_df, r_km)
    m_a100_excl = m_a100[~m_a100["shared_with_other_sink"]].copy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.2, 4.9), gridspec_kw={"wspace": 0.23})

    xlabel = r"$\log_{10} P_{\mathrm{mancha}}$ (População urbana CCA na bacia)"
    ylabel = rf"$\log_{{10}} \bar{{C}}$ (Média no mar, $R = {int(r_km)}$ km, $\mathrm{{pieces/m^3}}$)"

    _plotar_scatter_alometria(
        ax1,
        m_100,
        "urban_population",
        f"1. Todas as bacias h=0 (100%, R={int(r_km)} km)",
        cor_pts="#2b83ba",
        cor_reta="#334155",
        xlabel=xlabel,
        ylabel=ylabel,
        destacar_microbacias=True,
    )
    rotulo_painel(ax1, "(a)", dentro=False, x=-0.14, y=1.03)

    _plotar_scatter_alometria(
        ax2,
        m_a100,
        "urban_population",
        f"2. Bacias ≥ 100 km² (todos os círculos, R={int(r_km)} km)",
        cor_pts="#2b83ba",
        cor_reta="#1d4ed8",
        xlabel=xlabel,
        ylabel=ylabel,
        destacar_microbacias=False,
    )
    rotulo_painel(ax2, "(b)", dentro=False, x=-0.14, y=1.03)

    _plotar_scatter_alometria(
        ax3,
        m_a100_excl,
        "urban_population",
        f"3. Círculos exclusivos (s/ ponto dividido, R={int(r_km)} km)",
        cor_pts="#1b7837",
        cor_reta="#047857",
        xlabel=xlabel,
        ylabel=ylabel,
        destacar_microbacias=False,
    )
    rotulo_painel(ax3, "(c)", dentro=False, x=-0.14, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_lei_escala_todas_vs_maiores_vs_exclusivas"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


def gerar_figura_comparacao_raios_fixos(
    df_sens: pd.DataFrame,
    df_bacias_100: pd.DataFrame,
    ow_df: pd.DataFrame,
    df_best: pd.DataFrame | None = None,
) -> Path:
    """
    Gera figura 2x2 testando passo a passo todos os raios inteiros R = 1, 2, ..., 22 km:
      - Painel (a): Evolucao de N (bacias estimadas) e R² (%) ao longo de R = 1..22 km.
      - Painel (b): Melhor raio por ajuste puro R* (max R² com N >= 30 -> R = 5 km).
      - Painel (c): Melhor pico intermediario (N >= 150 -> R = 13 km).
      - Painel (d): Limite do mar territorial (R = 22 km, maxima cobertura N = 314 / 624).
    """
    aplicar_estilo()

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), gridspec_kw={"wspace": 0.24, "hspace": 0.30})
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    mask_base = (
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["r_km"].between(1, 22))
    )
    sub_all = df_sens[mask_base & (df_sens["metrica"] == "concentracao")].sort_values("r_km").copy()
    sub_flux = df_sens[mask_base & (df_sens["metrica"] == "fluxo")].sort_values("r_km").copy()
    sub_ex = df_sens[
        (df_sens["regime"] == "3_Bacias_Area_ge_100km2_Exclusivas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["metrica"] == "concentracao")
        & (df_sens["r_km"].between(1, 22))
    ].sort_values("r_km").copy()

    r_best_conc = 5
    r_best_trans = 13
    if df_best is not None and not df_best.empty:
        row_c = df_best[df_best["criterio"].str.startswith("Max R² — Urb. CCA Concentracao")]
        if not row_c.empty:
            r_best_conc = int(row_c.iloc[0]["r_otimo_km"])
        row_t = df_best[df_best["criterio"].str.startswith("Compromisso Intermediario")]
        if not row_t.empty:
            r_best_trans = int(row_t.iloc[0]["r_otimo_km"])

    r_vals = sub_all["r_km"].to_numpy()
    r2_all = sub_all["r2_pct"].to_numpy()
    r2_flux = sub_flux["r2_pct"].to_numpy()
    n_all = sub_all["n_bacias_estimadas"].to_numpy()
    r2_ex = sub_ex["r2_pct"].to_numpy()

    ax_a.axvspan(float(r_best_conc), 22.0, color="#d9f0d3", alpha=0.35, zorder=1, label=f"Faixa útil ({r_best_conc}–22 km)")
    ax_a.axvline(float(r_best_conc), color="#047857", linestyle="-.", linewidth=1.1, alpha=0.75, zorder=2)
    ax_a.plot(
        sub_flux["r_km"].to_numpy(),
        r2_flux,
        color="#7c3aed",
        marker="D",
        markersize=4.0,
        linewidth=1.5,
        zorder=4,
        label=r"$R^2$ (%) — Fluxo urbano ($\bar{C} \times A$)",
    )
    ax_a.plot(
        r_vals,
        r2_all,
        color="#1d4ed8",
        marker="o",
        markersize=4.6,
        linewidth=1.7,
        zorder=5,
        label=r"$R^2$ (%) — Concentração média ($\bar{C}$)",
    )
    ax_a.plot(
        sub_ex["r_km"].to_numpy(),
        r2_ex,
        color="#1b7837",
        marker="^",
        markersize=4.1,
        linewidth=1.2,
        linestyle="--",
        zorder=4,
        label=r"$R^2$ (%) — Círculos exclusivos",
    )

    ax_a.set_xlabel("Raio fixo R em todas as fozes: busca 1 a 22 km (km)", fontsize=9.6)
    ax_a.set_ylabel(r"Qualidade do ajuste $R^2$ (%)", fontsize=9.8, color="#1e3a8a")
    ax_a.tick_params(axis="y", labelcolor="#1e3a8a")
    ax_a.set_xlim(0.4, 22.8)
    ax_a.set_xticks(range(2, 23, 2))
    ax_a.set_ylim(-6, 56)
    ax_a.set_title("Varredura completa R = 1..22 km (cobertura N vs. ajuste R²)", fontsize=10.2, pad=9)

    ax_a_r = ax_a.twinx()
    ax_a_r.plot(
        r_vals,
        n_all,
        color="#b91c1c",
        marker="s",
        markersize=4.0,
        linewidth=1.4,
        linestyle=":",
        zorder=4,
        label="Nº bacias urbanas estimadas (N)",
    )
    ax_a_r.set_ylabel("Bacias urbanas estimadas (N)", fontsize=9.6, color="#b91c1c")
    ax_a_r.tick_params(axis="y", labelcolor="#b91c1c", direction="out", length=4.0, width=0.8)
    ax_a_r.set_ylim(-40, 620)

    anotacoes_a = [
        (r_best_conc, (8.4, 49.0), "Ótimo "),
        (r_best_trans, (10.6, 28.5), "Interm. "),
        (22, (19.0, 2.2), ""),
    ]
    for r_pt, (tx, ty), prefixo in anotacoes_a:
        row_r = sub_all[sub_all["r_km"] == r_pt].iloc[0]
        r2_str = f"{row_r['r2_pct']:.1f}%".replace(".", ",")
        ax_a.annotate(
            f"{prefixo}R={r_pt} km: N={int(row_r['n_bacias_estimadas'])}, R²={r2_str}",
            xy=(r_pt, row_r["r2_pct"]),
            xytext=(tx, ty),
            fontsize=6.9,
            ha="center",
            va="center",
            color="#1e3a8a",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#93c5fd", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=0.75),
            zorder=9,
        )

    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a_r.get_legend_handles_labels()
    ax_a.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        fontsize=6.6,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
    )
    rotulo_painel(ax_a, "(a)", dentro=False, x=-0.13, y=1.03)

    df_a100 = df_bacias_100[df_bacias_100["area_km2"] >= 100.0].copy().reset_index(drop=True)
    xlabel = r"$\log_{10} P_{\mathrm{mancha}}$ (População urbana CCA na bacia)"

    configs_painel = [
        (ax_b, "(b)", r_best_conc, f"Melhor ajuste puro R* = {r_best_conc} km (pluma imediata)", "#1b7837", "#047857"),
        (ax_c, "(c)", r_best_trans, f"Melhor pico intermediário R = {r_best_trans} km", "#0284c7", "#0369a1"),
        (ax_d, "(d)", 22, "Mar territorial R = 22 km (máxima cobertura)", "#b91c1c", "#7f1d1d"),
    ]
    for ax_p, lbl_p, r_val, tit, c_pt, c_ln in configs_painel:
        m_a100 = calcular_media_raio_fixo_nos_sinks(df_a100, ow_df, float(r_val))
        _plotar_scatter_alometria(
            ax_p,
            m_a100,
            "urban_population",
            tit,
            cor_pts=c_pt,
            cor_reta=c_ln,
            xlabel=xlabel,
            ylabel=rf"$\log_{{10}} \bar{{C}}$ (Média no mar em $R = {r_val}$ km, $\mathrm{{pieces/m^3}}$)",
            destacar_microbacias=False,
        )
        rotulo_painel(ax_p, lbl_p, dentro=False, x=-0.13, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_comparacao_raios_fixos_lei_escala"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


def gerar_figura_barras_melhor_raio_1_a_22km(
    df_sens: pd.DataFrame,
    df_best: pd.DataFrame | None = None,
) -> Path:
    """
    Gera um grafico de barras direto e simples (R = 1..22 km) destacando qual foi o R escolhido:
      - Painel superior (a): Barras de R² (%) por raio fixo R = 1..22 km (pico absoluto em R* = 5 km).
      - Painel inferior (b): Barras do numero de bacias estimadas (N) para cada raio R = 1..22 km.
    """
    aplicar_estilo()

    sub = df_sens[
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["metrica"] == "concentracao")
        & (df_sens["r_km"].between(1, 22))
    ].sort_values("r_km").copy()

    sub_flux = df_sens[
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["metrica"] == "fluxo")
        & (df_sens["r_km"].between(1, 22))
    ].sort_values("r_km").copy()

    r_best = 5
    r_trans = 13
    if df_best is not None and not df_best.empty:
        rc = df_best[df_best["criterio"].str.startswith("Max R² — Urb. CCA Concentracao")]
        if not rc.empty:
            r_best = int(rc.iloc[0]["r_otimo_km"])
        rt = df_best[df_best["criterio"].str.startswith("Compromisso Intermediario")]
        if not rt.empty:
            r_trans = int(rt.iloc[0]["r_otimo_km"])

    raios = sub["r_km"].to_numpy(dtype=int)
    n_vals = sub["n_bacias_estimadas"].to_numpy(dtype=int)
    r2_vals = np.where(np.isfinite(sub["r2_pct"].to_numpy()), sub["r2_pct"].to_numpy(), 0.0)
    p_vals = sub["p_value"].to_numpy()

    cores = []
    bordas = []
    lws = []
    hatches = []
    for r_k, n_k, p_k in zip(raios, n_vals, p_vals):
        if r_k == r_best:
            cores.append("#059669")
            bordas.append("#064e3b")
            lws.append(1.4)
            hatches.append("")
        elif r_k == r_trans:
            cores.append("#2563eb")
            bordas.append("#1e3a8a")
            lws.append(1.0)
            hatches.append("")
        elif r_k == 22:
            cores.append("#dc2626")
            bordas.append("#7f1d1d")
            lws.append(1.0)
            hatches.append("")
        elif n_k < 30 or not np.isfinite(p_k) or p_k >= 0.05:
            cores.append("#e5e7eb")
            bordas.append("#9ca3af")
            lws.append(0.7)
            hatches.append("///")
        else:
            cores.append("#cbd5e1")
            bordas.append("#475569")
            lws.append(0.65)
            hatches.append("")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12.8, 6.9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.16},
    )

    bars1 = ax1.bar(raios, r2_vals, width=0.70, zorder=3)
    for b, c, ec, lw, ht in zip(bars1, cores, bordas, lws, hatches):
        b.set_facecolor(c)
        b.set_edgecolor(ec)
        b.set_linewidth(lw)
        if ht:
            b.set_hatch(ht)

    for r_k, n_k, r2_v, p_k in zip(raios, n_vals, r2_vals, p_vals):
        if n_k < 4:
            ax1.text(r_k, 0.7, "sem\najuste", ha="center", va="bottom", fontsize=6.4, color="#6b7280")
        else:
            txt = f"{r2_v:.1f}%".replace(".", ",")
            cor_txt = "#064e3b" if r_k == r_best else ("#1e3a8a" if r_k == r_trans else ("#7f1d1d" if r_k == 22 else "#334155"))
            peso = "bold" if r_k in (r_best, r_trans, 22) else "normal"
            if n_k < 30:
                txt += "\n(n.s.)"
            ax1.text(r_k, r2_v + 0.65, txt, ha="center", va="bottom", fontsize=7.0, fontweight=peso, color=cor_txt)

    row_best = sub[sub["r_km"] == r_best].iloc[0]
    row_best_f = sub_flux[sub_flux["r_km"] == r_best].iloc[0]
    ax1.annotate(
        f"ESCOLHIDO: R* = {r_best} km (melhor resultado)\n"
        f"• Concentração: R² = {row_best['r2_pct']:.1f}% (β = {row_best['beta']:+.2f}, N = {int(row_best['n_bacias_estimadas'])})\n"
        f"• Com Fluxo (C̄ × A): R² = {row_best_f['r2_pct']:.1f}% (β = {row_best_f['beta']:+.2f})".replace(".", ","),
        xy=(r_best + 0.36, float(row_best["r2_pct"]) - 2.8),
        xytext=(6.6, 36.8),
        fontsize=7.6,
        fontweight="bold",
        color="#064e3b",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#ecfdf5", edgecolor="#059669", linewidth=1.25),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5, connectionstyle="arc3,rad=-0.10"),
        zorder=9,
    )

    ax1.set_ylabel(r"Força da Lei de Potência $R^2$ (%)", fontsize=10.0)
    ax1.set_ylim(0, 45.5)
    ax1.set_title("Escolha do melhor raio fixo (R = 1 a 22 km) — Mancha Urbana CCA vs. Média no Mar", fontsize=11.0, pad=10)
    leg_elems = [
        Patch(facecolor="#059669", edgecolor="#064e3b", lw=1.2, label=f"R* = {r_best} km — Escolhido (melhor R²)"),
        Patch(facecolor="#2563eb", edgecolor="#1e3a8a", lw=0.9, label=f"R = {r_trans} km — Pico interm. (N = 157)"),
        Patch(facecolor="#dc2626", edgecolor="#7f1d1d", lw=0.9, label="R = 22 km — Mar territ. (N = 314)"),
        Patch(facecolor="#cbd5e1", edgecolor="#475569", lw=0.6, label="Válidos (N ≥ 30, p < 0,05)"),
        Patch(facecolor="#e5e7eb", edgecolor="#9ca3af", lw=0.6, hatch="///", label="Amostra pequena / n.s. (R ≤ 3 km)"),
    ]
    ax1.legend(handles=leg_elems, loc="upper right", fontsize=7.2, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    rotulo_painel(ax1, "(a)", dentro=False, x=-0.06, y=1.03)

    bars2 = ax2.bar(raios, n_vals, width=0.70, zorder=3)
    for b, c, ec, lw, ht in zip(bars2, cores, bordas, lws, hatches):
        b.set_facecolor(c)
        b.set_edgecolor(ec)
        b.set_linewidth(lw)
        if ht:
            b.set_hatch(ht)

    for r_k, n_k in zip(raios, n_vals):
        peso = "bold" if r_k in (r_best, r_trans, 22) else "normal"
        cor_txt = "#064e3b" if r_k == r_best else ("#1e3a8a" if r_k == r_trans else ("#7f1d1d" if r_k == 22 else "#334155"))
        ax2.text(r_k, n_k + 6.0, str(n_k), ha="center", va="bottom", fontsize=7.3, fontweight=peso, color=cor_txt)

    ax2.set_xticks(raios)
    ax2.set_xticklabels([f"{r}" for r in raios], fontsize=9.0)
    ax2.set_xlim(0.2, 22.8)
    ax2.set_ylim(0, 365)
    ax2.set_xlabel("Raio fixo R aplicado em todas as fozes (km)", fontsize=10.2)
    ax2.set_ylabel("Bacias urbanas estimadas (N)", fontsize=10.0)
    rotulo_painel(ax2, "(b)", dentro=False, x=-0.06, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_barras_melhor_raio_1_a_22km"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


def gerar_figura_scatter_por_tipo_lixo(
    df_bacias_100: pd.DataFrame,
    df_noaa_all: pd.DataFrame,
    ow_sea: pd.DataFrame,
    r_km: float = 22.0,
) -> Path:
    """
    Gera 1 grafico de dispersao (Pop da bacia/mancha vs Media no circulo de raio R = 22 km)
    para cada um dos 4 tipos de lixo da base NOAA sobre as bacias >= 100 km² do GebcoCutter.
    """
    aplicar_estilo()

    df_a100 = df_bacias_100[df_bacias_100["area_km2"] >= 100.0].copy().reset_index(drop=True)

    tipos = [
        (
            "Ocean water mar (0,02–5,0 mm)",
            ow_sea,
            "urban_population",
            r"$\log_{10} P_{\mathrm{mancha}}$ (População urbana CCA)",
            r"$\log_{10} \bar{C}$ ($\mathrm{pieces/m^3}$)",
            "#2b83ba",
            "#1e3a8a",
        ),
        (
            "Pellets industriais (Nurdle Patrol)",
            df_noaa_all[df_noaa_all["Medium"] == "Nurdle Patrol"].copy(),
            "urban_population",
            r"$\log_{10} P_{\mathrm{mancha}}$ (População urbana CCA)",
            r"$\log_{10} \bar{M}$ ($\mathrm{pieces/10\ mins}$)",
            "#525252",
            "#1f2937",
        ),
        (
            "Areia de praia (Beach)",
            df_noaa_all[(df_noaa_all["Medium"] == "Beach") & (df_noaa_all["Unit"] == "pieces/m3")].copy(),
            "total_population",
            r"$\log_{10} P_{\mathrm{bacia}}$ (População total na bacia)",
            r"$\log_{10} \bar{C}$ ($\mathrm{pieces/m^3}$ areia)",
            "#d7191c",
            "#7f1d1d",
        ),
        (
            "Sedimento marinho (Ocean sediment)",
            df_noaa_all[df_noaa_all["Medium"] == "Ocean sediment"].copy(),
            "total_population",
            r"$\log_{10} P_{\mathrm{bacia}}$ (População total na bacia)",
            r"$\log_{10} \bar{C}$ ($\mathrm{pieces\ kg^{-1}\ d.w.}$)",
            "#7b3294",
            "#4c1d95",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6), gridspec_kw={"wspace": 0.24, "hspace": 0.30})
    paineis = [(axes[0, 0], "(a)"), (axes[0, 1], "(b)"), (axes[1, 0], "(c)"), (axes[1, 1], "(d)")]

    for (ax, lbl_p), (nome_t, sub_pts, pop_col, xl, yl, c_pt, c_ln) in zip(paineis, tipos):
        m_df = calcular_media_raio_fixo_nos_sinks(df_a100, sub_pts, r_km)
        _plotar_scatter_alometria(
            ax,
            m_df,
            pop_col,
            f"{nome_t} — R = {int(r_km)} km (bacias ≥ 100 km²)",
            cor_pts=c_pt,
            cor_reta=c_ln,
            xlabel=xl,
            ylabel=yl,
            destacar_microbacias=False,
        )
        rotulo_painel(ax, lbl_p, dentro=False, x=-0.13, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_scatter_por_tipo_lixo_media_raio"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


if __name__ == "__main__":
    print("1. Carregando Agua Oceanica (0,02-5,0 mm) c/ filtro de mar (GEBCO h <= 0)...")
    _, ow_sea = carregar_ocean_water_filtrado_mar()
    df_noaa = carregar_base_noaa()
    print(f"   Agua Oceanica no mar (h <= 0): {len(ow_sea)} pontos")

    print("2. Carregando catalogo global de bacias (externos/GebcoCutter + IPBA + CCA)...")
    df_bacias_100 = extrair_ou_carregar_bacias_100pct(ow_sea)
    n_tot_b = len(df_bacias_100)
    n_a100_b = int((df_bacias_100["area_km2"] >= 100.0).sum())
    print(f"   Total de bacias (porcoes ativas): {n_tot_b} | Bacias >= 100 km2: {n_a100_b}")

    print("3. Testando TODOS os raios fixos inteiros R = 1, 2, ..., 22 km em busca do melhor resultado...")
    df_sens = avaliar_sensibilidade_raios_lei_escala(df_bacias_100, ow_sea, raios=list(range(1, 23)))
    sens_csv = REPO_ROOT / "relatorios" / "lei_escala_sensibilidade_raio_fixo.csv"
    df_sens.to_csv(sens_csv, index=False)

    df_best = selecionar_melhores_raios_1_a_22km(df_sens, min_n=30)
    best_csv = REPO_ROOT / "relatorios" / "lei_escala_melhor_raio_1_a_22km.csv"
    df_best.to_csv(best_csv, index=False)
    print(f"   Tabela completa R = 1..22 km salva em: {sens_csv}")
    print(f"   Ranking dos melhores raios em 1..22 km salvo em: {best_csv}")

    print("\nVarredura completa R = 1..22 km (Pop. Urbana CCA vs. Concentracao Media no mar, Bacias >= 100 km2):")
    sub_print = df_sens[
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["metrica"] == "concentracao")
    ].sort_values("r_km")
    print(
        sub_print[["r_km", "n_bacias_com_pts_mar", "n_bacias_estimadas", "beta", "se", "r", "r2_pct", "p_value", "spearman_rho"]].to_string(
            index=False
        )
    )

    print("\nMelhores raios selecionados automaticamente em [1..22 km]:")
    print(df_best.to_string(index=False))

    row_urb_conc = df_best[df_best["criterio"].str.startswith("Max R² — Urb. CCA Concentracao")]
    r_otimo = float(row_urb_conc.iloc[0]["r_otimo_km"]) if not row_urb_conc.empty else 5.0

    print("\n4. Gerando as 4 figuras cientificas...")
    f1 = gerar_figura_todas_vs_maiores_vs_exclusivas(df_bacias_100, ow_sea, r_km=r_otimo)
    f2 = gerar_figura_comparacao_raios_fixos(df_sens, df_bacias_100, ow_sea, df_best=df_best)
    f3 = gerar_figura_scatter_por_tipo_lixo(df_bacias_100, df_noaa, ow_sea, r_km=22.0)
    f4 = gerar_figura_barras_melhor_raio_1_a_22km(df_sens, df_best=df_best)
    print(f"   Figuras salvas em:\n     - {f1}\n     - {f2}\n     - {f3}\n     - {f4}")
