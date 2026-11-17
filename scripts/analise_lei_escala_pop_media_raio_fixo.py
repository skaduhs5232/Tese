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
    out["shared_with_other_sink"] = shared
    out["max_point_usage"] = max_use
    return out


def _ajustar_ols(df_sub: pd.DataFrame, pop_col: str, y_col: str = "mp_mean") -> dict[str, Any]:
    valid = df_sub[(df_sub[pop_col] > 0) & (df_sub[y_col] > 0) & np.isfinite(df_sub[y_col])].copy()
    n = len(valid)
    if n < 4:
        return {
            "N": n,
            "beta": np.nan,
            "se": np.nan,
            "r": np.nan,
            "r2": np.nan,
            "p_val": np.nan,
            "rho": np.nan,
            "p_rho": np.nan,
        }
    x = np.log10(valid[pop_col].to_numpy())
    y = np.log10(valid[y_col].to_numpy())
    sl, ic, rv, pv, se = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)
    return {
        "N": int(n),
        "beta": float(sl),
        "intercept": float(ic),
        "se": float(se),
        "r": float(rv),
        "r2": float(rv**2),
        "p_val": float(pv),
        "rho": float(rho),
        "p_rho": float(prho),
    }


def avaliar_sensibilidade_raios_lei_escala(
    df_bacias_100: pd.DataFrame,
    ow_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Avalia para cada raio fixo R = 1..50 km quantas bacias sao estimadas (N)
    e qual a correlacao (beta, R², r, p-valor) nos regimes:
      1. Todas as bacias (100% dos sinks h = 0, sem corte de area)
      2. Bacias principais (Area >= 100 km², todas)
      3. Bacias principais com circulos exclusivos (`shared_with_other_sink == False`)
      4. Top 95% global de area acumulada
    """
    raios = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 22, 25, 30, 40, 50]
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
                res = _ajustar_ols(df_reg, pop_col, "mp_mean")
                rows.append(
                    {
                        "r_km": r_km,
                        "regime": regime_nome,
                        "pop_col": pop_col,
                        "n_bacias_estimadas": res["N"],
                        "beta": res["beta"],
                        "se": res["se"],
                        "r": res["r"],
                        "r2_pct": res["r2"] * 100.0 if np.isfinite(res["r2"]) else np.nan,
                        "p_value": res["p_val"],
                        "spearman_rho": res["rho"],
                        "spearman_p": res["p_rho"],
                    }
                )
    return pd.DataFrame(rows)


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
) -> Path:
    """
    Gera figura 2x2 respondendo a pergunta do orientador:
    "Para raio R1 voce conseguiu estimar para x bacias, para raio R2 para 2x... qual o melhor valor?"
      - Painel (a): Evolucao de N (bacias estimadas) e R² (%) para Area >= 100 km² em funcao de R.
      - Painel (b): Scatter Lei de Escala para R = 5 km (pico de R² e beta ~ 0,70).
      - Painel (c): Scatter Lei de Escala para R = 10 km (atalho superior do Audio 4).
      - Painel (d): Scatter Lei de Escala para R = 22 km (mar territorial, maxima cobertura N).
    """
    aplicar_estilo()

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6), gridspec_kw={"wspace": 0.24, "hspace": 0.30})
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    sub_all = df_sens[
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["r_km"] >= 2)
    ].copy()
    sub_ex = df_sens[
        (df_sens["regime"] == "3_Bacias_Area_ge_100km2_Exclusivas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["r_km"] >= 2)
    ].copy()

    r_vals = sub_all["r_km"].to_numpy()
    r2_all = sub_all["r2_pct"].to_numpy()
    n_all = sub_all["n_bacias_estimadas"].to_numpy()
    r2_ex = sub_ex["r2_pct"].to_numpy()

    ax_a.axvspan(5.0, 22.0, color="#d9f0d3", alpha=0.40, zorder=1, label="Faixa de compromisso (5–22 km)")
    ax_a.plot(
        r_vals,
        r2_all,
        color="#1d4ed8",
        marker="o",
        markersize=4.8,
        linewidth=1.7,
        zorder=5,
        label=r"$R^2$ (%) — Bacias ≥ 100 km² (todos círculos)",
    )
    ax_a.plot(
        sub_ex["r_km"].to_numpy(),
        r2_ex,
        color="#1b7837",
        marker="^",
        markersize=4.4,
        linewidth=1.3,
        linestyle="--",
        zorder=4,
        label=r"$R^2$ (%) — Círculos exclusivos",
    )

    ax_a.set_xlabel("Raio fixo R do círculo na foz (km)", fontsize=9.8)
    ax_a.set_ylabel(r"Qualidade do ajuste $R^2$ (%)", fontsize=9.8, color="#1e3a8a")
    ax_a.tick_params(axis="y", labelcolor="#1e3a8a")
    ax_a.set_xlim(0, 52)
    ax_a.set_ylim(-4, 46)
    ax_a.set_title("Escolha do raio fixo R (cobertura N vs. ajuste R²)", fontsize=10.3, pad=9)

    ax_a_r = ax_a.twinx()
    ax_a_r.plot(
        r_vals,
        n_all,
        color="#b91c1c",
        marker="s",
        markersize=4.3,
        linewidth=1.4,
        linestyle=":",
        zorder=4,
        label="Nº de bacias urbanas estimadas (N)",
    )
    ax_a_r.set_ylabel("Bacias urbanas estimadas (N)", fontsize=9.6, color="#b91c1c")
    ax_a_r.tick_params(axis="y", labelcolor="#b91c1c", direction="out", length=4.0, width=0.8)
    ax_a_r.set_ylim(0, 720)

    anotacoes_a = [
        (5, (15.5, 35.0)),
        (10, (14.5, 24.5)),
        (22, (34.0, 11.5)),
    ]
    for r_pt, (tx, ty) in anotacoes_a:
        row_r = sub_all[sub_all["r_km"] == r_pt].iloc[0]
        r2_str = f"{row_r['r2_pct']:.1f}%".replace(".", ",")
        ax_a.annotate(
            f"R={r_pt} km: N={int(row_r['n_bacias_estimadas'])}, R²={r2_str}",
            xy=(r_pt, row_r["r2_pct"]),
            xytext=(tx, ty),
            fontsize=7.2,
            ha="center",
            va="center",
            color="#1e3a8a",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#93c5fd", alpha=0.94),
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=0.75),
            zorder=9,
        )

    ax_a.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.92)
    rotulo_painel(ax_a, "(a)", dentro=False, x=-0.13, y=1.03)

    df_a100 = df_bacias_100[df_bacias_100["area_km2"] >= 100.0].copy().reset_index(drop=True)
    xlabel = r"$\log_{10} P_{\mathrm{mancha}}$ (População urbana CCA na bacia)"

    configs_painel = [
        (ax_b, "(b)", 5, "Atalho R = 5 km (pluma imediata)", "#1b7837", "#047857"),
        (ax_c, "(c)", 10, "Atalho R = 10 km (transição costeira)", "#0284c7", "#0369a1"),
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

    print("3. Avaliando sensibilidade do raio fixo R (1 a 50 km) para a Lei de Escala...")
    df_sens = avaliar_sensibilidade_raios_lei_escala(df_bacias_100, ow_sea)
    sens_csv = REPO_ROOT / "relatorios" / "lei_escala_sensibilidade_raio_fixo.csv"
    df_sens.to_csv(sens_csv, index=False)
    print(f"   Tabela de sensibilidade salva em: {sens_csv}")

    print("\nResumo da Lei de Escala (Pop. Urbana da Mancha vs. Media no mar, Bacias >= 100 km2):")
    sub_print = df_sens[
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["r_km"].isin([2, 3, 5, 8, 10, 12, 15, 20, 22, 25, 30, 50]))
    ]
    print(
        sub_print[["r_km", "n_bacias_estimadas", "beta", "se", "r", "r2_pct", "p_value", "spearman_rho"]].to_string(
            index=False
        )
    )

    print("\n4. Gerando as 3 figuras cientificas...")
    f1 = gerar_figura_todas_vs_maiores_vs_exclusivas(df_bacias_100, ow_sea, r_km=5.0)
    f2 = gerar_figura_comparacao_raios_fixos(df_sens, df_bacias_100, ow_sea)
    f3 = gerar_figura_scatter_por_tipo_lixo(df_bacias_100, df_noaa, ow_sea, r_km=22.0)
    print(f"   Figuras salvas em:\n     - {f1}\n     - {f2}\n     - {f3}")
