"""
scripts/analise_voronoi_e_diluicao_mesmo_espaco.py
---------------------------------------------------
Investiga os dois refinamentos metodologicos apontados na Licao 03 sobre a branch `sinksRaioNoaa`:

1. Item 1 — Sobreposicao de fozes vizinhas (Voronoi R = 1..22 km):
   Atribui cada ponto de coleta no mar (`GEBCO h <= 0`) EXCLUSIVAMENTE a 1 unica foz mais proxima
   (evitando a multiplicacao de pontos em ~14 circulos vizinhos em R = 22 km), em dois regimes:
     - Voronoi Global (entre todas as 179.718 fozes IPBA, filtrando A >= 100 km² apos atribuicao)
     - Voronoi Bacias Principais (competicao apenas entre bacias com A >= 100 km²)

2. Item 2 — Diluicao pela agua do rio no "mesmo espaço":
   Avalia Concentracao Media (`pieces/m3`) e Fluxo (`C_bar * A_bacia`) de forma estritamente
   comparavel no mesmo espaco populacional e amostral:
     - Mesmo conjunto de populacao (`urban_population` e `total_population`) para R = 5, 13 e 22 km
     - Mesma coorte fixa de bacias acompanhada de R = 5 km ate R = 22 km (`N_comum`)
     - Espaco global unificado (100% das bacias sem corte de area)
     - Normalizacao para o mesmo espaco unitario de area: Densidade Populacional (P / A) vs C_bar,
       modelo conjunto `log10(C_bar) ~ alpha * log10(P) + gamma * log10(A)`,
       correlacao parcial `Partial(C_bar, P | A)` e concentracao equivalente em A_ref = 1.000 km².
"""

from __future__ import annotations

import sys
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
from scripts.analise_lei_escala_pop_media_raio_fixo import (
    _ajustar_ols,
    calcular_media_raio_fixo_nos_sinks,
)
from scripts.analise_r22km_mar_territorial import (
    carregar_ocean_water_filtrado_mar,
    carregar_ou_gerar_bacias_globais_externos,
    to_xyz,
)


def precomputar_vizinho_mais_proximo(
    bacias_df: pd.DataFrame,
    ow_sea: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula uma unica vez o sink mais proximo (Voronoi) e sua distancia em km."""
    s_xyz = to_xyz(bacias_df["latitude"].to_numpy(), bacias_df["longitude"].to_numpy())
    p_xyz = to_xyz(ow_sea["lat"].to_numpy(), ow_sea["lon"].to_numpy())
    tree = cKDTree(s_xyz)
    d_chord, idx_near = tree.query(p_xyz, k=1)
    dist_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(d_chord / 2.0, 0.0, 1.0))
    return idx_near, dist_km


def agregar_voronoi_no_raio(
    bacias_df: pd.DataFrame,
    ow_sea: pd.DataFrame,
    idx_near: np.ndarray,
    dist_km: np.ndarray,
    r_km: float,
) -> pd.DataFrame:
    """Agrega por media aritmetica os pontos cuja foz mais proxima esta a distancia <= r_km."""
    mask = dist_km <= float(r_km)
    tmp = pd.DataFrame({"bidx": idx_near[mask], "valor": ow_sea["valor"].to_numpy()[mask]})
    agg = (
        tmp.groupby("bidx", as_index=False)
        .agg(mp_mean=("valor", "mean"), mp_std=("valor", "std"), n_points=("valor", "count"))
    )
    out = bacias_df.copy().reset_index(drop=True)
    out["bidx"] = out.index
    out = out.merge(agg, on="bidx", how="left")
    out["n_points"] = out["n_points"].fillna(0).astype(int)
    out["mp_std"] = out["mp_std"].fillna(0.0)
    out["flux"] = out["mp_mean"] * out["area_km2"]
    return out


def executar_varredura_voronoi_1_a_22km(
    df_all: pd.DataFrame,
    ow_sea: pd.DataFrame,
) -> pd.DataFrame:
    """Executa a varredura de R = 1 a 22 km comparando Circulo Sobreposto vs Voronoi (A e B)."""
    a100 = df_all[df_all["area_km2"] >= 100.0].copy().reset_index(drop=True)
    idx_all, dist_all = precomputar_vizinho_mais_proximo(df_all, ow_sea)
    idx_a100, dist_a100 = precomputar_vizinho_mais_proximo(a100, ow_sea)

    rows: list[dict[str, Any]] = []
    for r_km in range(1, 23):
        m_ovl = calcular_media_raio_fixo_nos_sinks(a100, ow_sea, float(r_km))
        m_va = agregar_voronoi_no_raio(a100, ow_sea, idx_a100, dist_a100, float(r_km))
        m_vb_full = agregar_voronoi_no_raio(df_all, ow_sea, idx_all, dist_all, float(r_km))
        m_vb = m_vb_full[m_vb_full["area_km2"] >= 100.0].copy()

        for metodo, df_m in [
            ("Circulo_Sobreposto_A100", m_ovl),
            ("Voronoi_A_Entre_A100", m_va),
            ("Voronoi_B_Global_Sinks", m_vb),
        ]:
            for pop_col in ["urban_population", "total_population"]:
                for metrica, y_col in [("concentracao", "mp_mean"), ("fluxo", "flux")]:
                    res = _ajustar_ols(df_m, pop_col, y_col)
                    r2_pct = res["r2"] * 100.0 if np.isfinite(res["r2"]) else np.nan
                    r2_bins_pct = res["r2_bins"] * 100.0 if np.isfinite(res["r2_bins"]) else np.nan
                    rows.append(
                        {
                            "r_km": int(r_km),
                            "metodo": metodo,
                            "pop_col": pop_col,
                            "metrica": metrica,
                            "n_bacias_com_pts": res["N_com_pts"],
                            "n_bacias": res["N"],
                            "beta": res["beta"],
                            "se": res["se"],
                            "r": res["r"],
                            "r2_pct": r2_pct,
                            "p_value": res["p_val"],
                            "spearman_rho": res["rho"],
                            "r2_bins_pct": r2_bins_pct,
                        }
                    )
    return pd.DataFrame(rows)


def _metricas_diluicao_e_mesmo_espaco(
    df_sub: pd.DataFrame,
    pop_col: str,
    label_espaco: str,
    r_km: int,
    a_ref_km2: float = 1000.0,
) -> dict[str, Any]:
    """
    Calcula todas as estatisticas do Item 2 em um mesmo espaco:
      - Concentracao Media (`mp_mean`) vs P
      - Fluxo Total (`mp_mean * area_km2`) vs P
      - Densidade Populacional (`P / area_km2`) vs Concentracao Media (`mp_mean`)
      - Regressao multipla `log10(C_bar) ~ alpha * log10(P) + gamma * log10(A)`
      - Correlacao parcial `r(log10 C_bar, log10 P | log10 A)`
      - Concentracao normalizada para uma mesma area de referencia `A_ref = 1.000 km2`
    """
    sub = df_sub[(df_sub[pop_col] > 0) & (df_sub["mp_mean"] > 0) & (df_sub["area_km2"] > 0)].copy()
    n = len(sub)
    lp = np.log10(sub[pop_col].to_numpy())
    la = np.log10(sub["area_km2"].to_numpy())
    lc = np.log10(sub["mp_mean"].to_numpy())
    lf = np.log10((sub["mp_mean"] * sub["area_km2"]).to_numpy())
    ld = lp - la

    sl_c, _, r_c, p_c, se_c = stats.linregress(lp, lc)
    sl_f, _, r_f, p_f, se_f = stats.linregress(lp, lf)
    sl_d, _, r_d, p_d, se_d = stats.linregress(ld, lc)
    r_pa, _ = stats.pearsonr(lp, la)

    # Regressao multipla log10(C) ~ alpha * log10(P) + gamma * log10(A)
    X = np.column_stack([np.ones(n), lp, la])
    coef, *_ = np.linalg.lstsq(X, lc, rcond=None)
    alpha_p = float(coef[1])
    gamma_a = float(coef[2])
    r2_joint = 1.0 - np.sum((lc - X @ coef) ** 2) / np.sum((lc - np.mean(lc)) ** 2)

    # Correlacao parcial (removendo A linearmente em log10 de P e de C)
    res_c = lc - np.polyval(np.polyfit(la, lc, 1), la)
    res_p = lp - np.polyval(np.polyfit(la, lp, 1), la)
    r_part, p_part = stats.pearsonr(res_p, res_c)

    # Concentracao equivalente normalizada para A_ref = 1.000 km²: C_eq = C_bar * (A / A_ref)^(-gamma_a)
    lc_eq = lc - gamma_a * (la - np.log10(a_ref_km2))
    sl_eq, _, r_eq, p_eq, se_eq = stats.linregress(lp, lc_eq)

    return {
        "espaco_amostral": label_espaco,
        "r_km": int(r_km),
        "pop_col": pop_col,
        "N": int(n),
        "beta_conc": float(sl_c),
        "se_conc": float(se_c),
        "r2_conc_pct": float(r_c**2 * 100.0),
        "p_conc": float(p_c),
        "beta_fluxo": float(sl_f),
        "se_fluxo": float(se_f),
        "r2_fluxo_pct": float(r_f**2 * 100.0),
        "p_fluxo": float(p_f),
        "beta_densidade_P_A": float(sl_d),
        "se_densidade": float(se_d),
        "r2_densidade_P_A_pct": float(r_d**2 * 100.0),
        "p_densidade": float(p_d),
        "r_acoplamento_P_A": float(r_pa),
        "alpha_P_controlando_A": alpha_p,
        "gamma_diluicao_A": gamma_a,
        "r2_multiplo_P_e_A_pct": float(r2_joint * 100.0),
        "r_parcial_C_P_dado_A": float(r_part),
        "r2_parcial_C_P_dado_A_pct": float(r_part**2 * 100.0),
        "p_parcial": float(p_part),
        "beta_Ceq_1000km2": float(sl_eq),
        "r2_Ceq_1000km2_pct": float(r_eq**2 * 100.0),
        "p_Ceq_1000km2": float(p_eq),
    }


def executar_calculo_diluicao_mesmo_espaco(
    df_all: pd.DataFrame,
    ow_sea: pd.DataFrame,
) -> pd.DataFrame:
    """Consolida os 4 recortes de 'mesmo espaco' para R = 5, 13 e 22 km."""
    a100 = df_all[df_all["area_km2"] >= 100.0].copy().reset_index(drop=True)
    m5_a100 = calcular_media_raio_fixo_nos_sinks(a100, ow_sea, 5.0)
    ids_49 = set(m5_a100[(m5_a100["urban_population"] > 0) & (m5_a100["mp_mean"] > 0)]["sink_id"])
    ids_65 = set(m5_a100[(m5_a100["total_population"] > 0) & (m5_a100["mp_mean"] > 0)]["sink_id"])

    idx_all, dist_all = precomputar_vizinho_mais_proximo(df_all, ow_sea)

    rows: list[dict[str, Any]] = []
    for r_km in [5, 13, 22]:
        m_a100 = calcular_media_raio_fixo_nos_sinks(a100, ow_sea, float(r_km))
        m_100pct = calcular_media_raio_fixo_nos_sinks(df_all, ow_sea, float(r_km))
        m_vb = agregar_voronoi_no_raio(df_all, ow_sea, idx_all, dist_all, float(r_km))
        m_vb_a100 = m_vb[m_vb["area_km2"] >= 100.0].copy()

        for pcol in ["urban_population", "total_population"]:
            rows.append(_metricas_diluicao_e_mesmo_espaco(m_a100, pcol, "1_Bacias_A>=100km2", r_km))
            coorte_ids = ids_49 if pcol == "urban_population" else ids_65
            rows.append(
                _metricas_diluicao_e_mesmo_espaco(
                    m_a100[m_a100["sink_id"].isin(coorte_ids)],
                    pcol,
                    "2_Coorte_Comum_de_5km",
                    r_km,
                )
            )
            rows.append(_metricas_diluicao_e_mesmo_espaco(m_100pct, pcol, "3_Todas_Bacias_100pct_SemCorte", r_km))
            rows.append(_metricas_diluicao_e_mesmo_espaco(m_vb_a100, pcol, "4_Voronoi_Global_A>=100km2", r_km))

    return pd.DataFrame(rows)


def gerar_figura_voronoi_e_mesmo_espaco(
    df_vor: pd.DataFrame,
    df_dil: pd.DataFrame,
    df_all: pd.DataFrame,
    ow_sea: pd.DataFrame,
) -> Path:
    """
    Gera figura 2x2 sintetica reunindo:
      (a) Item 1 — Comparacao Voronoi Global vs. Circulo Sobreposto em R = 1..22 km (Urbano).
      (b) Item 1 + Item 2 — Voronoi Global em R = 22 km: Concentracao vs. Fluxo na mesma escala.
      (c) Item 2 no mesmo espaco amostral & populacional: R² Conc vs. Fluxo para 5, 13, 22 km e 100%.
      (d) Item 2 controle real de diluicao: Parcial (C,P|A) e Densidade (P/A) mantendo A fixo.
    """
    aplicar_estilo()
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.8), gridspec_kw={"wspace": 0.25, "hspace": 0.32})
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- Painel (a): Voronoi vs Sobreposto de R = 3..22 km ---
    for metodo, metrica, lbl, cor, ls, mk in [
        ("Voronoi_B_Global_Sinks", "fluxo", "Voronoi Global — Fluxo (C̄ × A)", "#047857", "-", "D"),
        ("Voronoi_B_Global_Sinks", "concentracao", "Voronoi Global — Concentração (C̄)", "#0284c7", "-", "o"),
        ("Voronoi_A_Entre_A100", "concentracao", "Voronoi Bacias ≥ 100 km² — Conc.", "#d97706", "--", "^"),
        ("Circulo_Sobreposto_A100", "concentracao", "Círculo Sobreposto — Conc.", "#b91c1c", ":", "s"),
    ]:
        sub = df_vor[
            (df_vor["metodo"] == metodo)
            & (df_vor["pop_col"] == "urban_population")
            & (df_vor["metrica"] == metrica)
            & (df_vor["r_km"] >= 3)
        ].sort_values("r_km")
        ax_a.plot(
            sub["r_km"],
            sub["r2_pct"],
            color=cor,
            linestyle=ls,
            marker=mk,
            markersize=4.2,
            linewidth=1.7,
            label=lbl,
        )

    ax_a.set_title("Item 1: Voronoi elimina a queda por sobreposição (Urb. CCA)", fontsize=10.0, pad=9)
    ax_a.set_xlabel("Raio fixo R de alcance no mar (km)", fontsize=9.6)
    ax_a.set_ylabel(r"Qualidade do ajuste $R^2$ (%)", fontsize=9.6)
    ax_a.set_xlim(2.5, 22.6)
    ax_a.set_xticks(range(4, 23, 2))
    ax_a.set_ylim(0, 54)
    ax_a.legend(loc="upper right", fontsize=6.9, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.94)
    rotulo_painel(ax_a, "(a)", dentro=False, x=-0.13, y=1.03)

    # --- Painel (b): Voronoi Global R = 22 km Concentracao vs Fluxo ---
    idx_all, dist_all = precomputar_vizinho_mais_proximo(df_all, ow_sea)
    vb_22 = agregar_voronoi_no_raio(df_all, ow_sea, idx_all, dist_all, 22.0)
    vb_22 = vb_22[(vb_22["area_km2"] >= 100.0) & (vb_22["urban_population"] > 0) & (vb_22["mp_mean"] > 0)].copy()

    xp = np.log10(vb_22["urban_population"].to_numpy())
    yc = np.log10(vb_22["mp_mean"].to_numpy())
    yf = np.log10(vb_22["flux"].to_numpy())
    # Centralizar ambos pela media apenas para comparar inclinacao e dispersao no mesmo quadro, ou mostrar eixo duplo
    ax_b_f = ax_b.twinx()

    ax_b.scatter(xp, yc, edgecolors="#0284c7", facecolors="none", s=34, linewidths=1.1, alpha=0.85, label="Concentração C̄ (esq.)")
    sl_c, ic_c, rv_c, pv_c, se_c = stats.linregress(xp, yc)
    xl = np.linspace(xp.min() - 0.1, xp.max() + 0.1, 100)
    ax_b.plot(xl, ic_c + sl_c * xl, color="#0369a1", linewidth=1.9, label=f"OLS Conc.: β={sl_c:+.2f}, R²={rv_c**2*100:.1f}%".replace(".", ","))

    ax_b_f.scatter(xp, yf, edgecolors="#047857", facecolors="none", marker="D", s=28, linewidths=1.0, alpha=0.72, label="Fluxo C̄×A (dir.)")
    sl_f, ic_f, rv_f, pv_f, se_f = stats.linregress(xp, yf)
    ax_b_f.plot(xl, ic_f + sl_f * xl, color="#047857", linewidth=1.9, linestyle="--", label=f"OLS Fluxo: β={sl_f:+.2f}, R²={rv_f**2*100:.1f}%".replace(".", ","))

    ax_b.set_title(f"Voronoi Global em R = 22 km (N = {len(vb_22)} bacias ≥ 100 km²)", fontsize=10.0, pad=9)
    ax_b.set_xlabel(r"$\log_{10} P_{\mathrm{mancha}}$ (População urbana CCA)", fontsize=9.6)
    ax_b.set_ylabel(r"$\log_{10} \bar{C}$ ($\mathrm{pieces/m^3}$)", fontsize=9.4, color="#0369a1")
    ax_b_f.set_ylabel(r"$\log_{10}(\bar{C} \cdot A)$ (Fluxo)", fontsize=9.4, color="#047857")
    ax_b.tick_params(axis="y", labelcolor="#0369a1")
    ax_b_f.tick_params(axis="y", labelcolor="#047857", direction="out", length=4.0, width=0.8)
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b_f.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.9, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.94)
    rotulo_painel(ax_b, "(b)", dentro=False, x=-0.13, y=1.03)

    # --- Painel (c): Item 2 no MESMO espaco amostral & populacional ---
    grupos = [
        ("R=5 km\nUrb (N=49)", "1_Bacias_A>=100km2", 5, "urban_population"),
        ("R=13 km\nUrb (N=157)", "1_Bacias_A>=100km2", 13, "urban_population"),
        ("R=22 km\nUrb (N=314)", "1_Bacias_A>=100km2", 22, "urban_population"),
        ("R=5 km\nTot (N=65)", "1_Bacias_A>=100km2", 5, "total_population"),
        ("R=22 km\nTot (N=624)", "1_Bacias_A>=100km2", 22, "total_population"),
        ("100% Global\n22km (N=8714)", "3_Todas_Bacias_100pct_SemCorte", 22, "total_population"),
    ]
    x_g = np.arange(len(grupos))
    r2_c_vals = []
    r2_f_vals = []
    for _, esp, r_k, pc in grupos:
        row = df_dil[(df_dil["espaco_amostral"] == esp) & (df_dil["r_km"] == r_k) & (df_dil["pop_col"] == pc)].iloc[0]
        r2_c_vals.append(row["r2_conc_pct"])
        r2_f_vals.append(row["r2_fluxo_pct"])

    w = 0.36
    b1 = ax_c.bar(x_g - w / 2, r2_c_vals, width=w, color="#60a5fa", edgecolor="black", lw=0.6, label=r"Concentração média $\bar{C}$")
    b2 = ax_c.bar(x_g + w / 2, r2_f_vals, width=w, color="#059669", edgecolor="black", lw=0.6, label=r"Fluxo ($\bar{C} \times A_{\mathrm{bacia}}$)")
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax_c.text(rect.get_x() + rect.get_width() / 2, h + 0.8, f"{h:.1f}%".replace(".", ","), ha="center", va="bottom", fontsize=6.6)

    ax_c.set_xticks(x_g)
    ax_c.set_xticklabels([g[0] for g in grupos], fontsize=7.6)
    ax_c.set_ylim(0, 50)
    ax_c.set_ylabel(r"$R^2$ (%)", fontsize=9.6)
    ax_c.set_title("Item 2 no mesmo espaço: ganho real Concentração vs. Fluxo", fontsize=10.0, pad=9)
    ax_c.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.94)
    rotulo_painel(ax_c, "(c)", dentro=False, x=-0.13, y=1.03)

    # --- Painel (d): Controle puro de diluicao (Area fixa / Densidade P/A / Voronoi) ---
    sub_d = [
        ("R=22 km\nTot (N=624)", "1_Bacias_A>=100km2", 22, "total_population"),
        ("R=13 km\nTot (N=300)", "1_Bacias_A>=100km2", 13, "total_population"),
        ("R=5 km\nUrb (N=49)", "1_Bacias_A>=100km2", 5, "urban_population"),
        ("Coorte 5→22\nUrb (N=50)", "2_Coorte_Comum_de_5km", 22, "urban_population"),
        ("Voronoi 22km\nUrb (N=44)", "4_Voronoi_Global_A>=100km2", 22, "urban_population"),
    ]
    xd = np.arange(len(sub_d))
    v_conc = []
    v_dens = []
    v_part = []
    v_flux = []
    for _, esp, r_k, pc in sub_d:
        row = df_dil[(df_dil["espaco_amostral"] == esp) & (df_dil["r_km"] == r_k) & (df_dil["pop_col"] == pc)].iloc[0]
        v_conc.append(row["r2_conc_pct"])
        v_dens.append(row["r2_densidade_P_A_pct"])
        v_part.append(row["r2_parcial_C_P_dado_A_pct"])
        v_flux.append(row["r2_fluxo_pct"])

    w2 = 0.20
    ax_d.bar(xd - 1.5 * w2, v_conc, width=w2, color="#93c5fd", edgecolor="black", lw=0.5, label="C̄ bruto vs P")
    ax_d.bar(xd - 0.5 * w2, v_dens, width=w2, color="#f59e0b", edgecolor="black", lw=0.5, label="C̄ vs Densidade (P/A)")
    ax_d.bar(xd + 0.5 * w2, v_part, width=w2, color="#7c3aed", edgecolor="black", lw=0.5, label="Parcial C̄ vs P | A fixa")
    ax_d.bar(xd + 1.5 * w2, v_flux, width=w2, color="#059669", edgecolor="black", lw=0.5, label="Fluxo (C̄ × A) vs P")

    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([s[0] for s in sub_d], fontsize=7.6)
    ax_d.set_ylim(0, 50)
    ax_d.set_ylabel(r"$R^2$ (%)", fontsize=9.6)
    ax_d.set_title("Mesmo espaço geométrico: Densidade P/A, Parcial | A e Voronoi", fontsize=10.0, pad=9)
    ax_d.legend(loc="upper left", ncol=2, fontsize=6.7, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.94)
    rotulo_painel(ax_d, "(d)", dentro=False, x=-0.13, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_voronoi_e_diluicao_mesmo_espaco"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


if __name__ == "__main__":
    print("1. Carregando Agua Oceanica filtrada no mar (GEBCO h <= 0) e bacias externas/...")
    _, ow_sea = carregar_ocean_water_filtrado_mar()
    df_all = carregar_ou_gerar_bacias_globais_externos(ow_sea)
    print(f"   Pontos no mar: {len(ow_sea)} | Total bacias IPBA: {len(df_all)}")

    print("\n2. Executando varredura Voronoi (1..22 km) vs Circulos Sobrepostos (Item 1)...")
    df_vor = executar_varredura_voronoi_1_a_22km(df_all, ow_sea)
    vor_csv = REPO_ROOT / "relatorios" / "voronoi_sensibilidade_1_a_22km.csv"
    df_vor.to_csv(vor_csv, index=False)
    print(f"   Salvo em: {vor_csv}")

    resumo_vor = df_vor[
        (df_vor["r_km"].isin([5, 10, 13, 18, 22]))
        & (df_vor["pop_col"] == "urban_population")
    ]
    print(resumo_vor[["r_km", "metodo", "metrica", "n_bacias", "beta", "r2_pct", "p_value"]].to_string(index=False))

    print("\n3. Executando calculo de diluicao no mesmo espaco (Item 2)...")
    df_dil = executar_calculo_diluicao_mesmo_espaco(df_all, ow_sea)
    dil_csv = REPO_ROOT / "relatorios" / "diluicao_fluxo_mesmo_espaco.csv"
    df_dil.to_csv(dil_csv, index=False)
    print(f"   Salvo em: {dil_csv}")
    cols_show = [
        "espaco_amostral", "r_km", "pop_col", "N",
        "r2_conc_pct", "r2_fluxo_pct", "r2_densidade_P_A_pct",
        "r2_parcial_C_P_dado_A_pct", "gamma_diluicao_A", "r2_Ceq_1000km2_pct",
    ]
    print(df_dil[cols_show].round(2).to_string(index=False))

    print("\n4. Gerando figura sintetica...")
    fig_path = gerar_figura_voronoi_e_mesmo_espaco(df_vor, df_dil, df_all, ow_sea)
    print(f"   Figura salva em: {fig_path}")
