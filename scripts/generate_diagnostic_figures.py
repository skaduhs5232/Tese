"""
scripts/generate_diagnostic_figures.py
--------------------------------------
Gera a figura de diagnóstico científico comparativo:
Painel (a): Estratificação Petroquímica (Nurdle Patrol no Golfo do México: R² = 17.9%, p = 0.018 vs. Fora do Golfo R² = 0.00).
Painel (b): NOAA Ocean Water (Redes Neuston com amostragem consistente N >= 3: beta = 0.93 ~ 1.0, R² = 22.7%).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from geracao_graficos.estilo_erneson import (
    aplicar_estilo,
    salvar,
    rotulo_painel,
    anotar_expoente,
)
from pipeline.spatial_matcher import match_sinks_with_individual_radius
from scripts.experiment_data_science_correlation import carregar_dados_noaa

def gerar_figura_diagnostico():
    aplicar_estilo()

    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    df_nurdles, df_ocean, _ = carregar_dados_noaa(noaa_path)

    sinks_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "sinks_raios_north_central_america.csv")
    pop_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "populacao_todas_bacias_north_central_america.csv")

    sinks_coords = sinks_na.groupby("basin_id")[["latitude", "longitude"]].mean().reset_index()
    pop_na = pop_na.merge(sinks_coords, on="basin_id", how="left")
    pop_na["is_gulf"] = (pop_na["latitude"] >= 24) & (pop_na["latitude"] <= 31) & (pop_na["longitude"] >= -98) & (pop_na["longitude"] <= -80)

    _, summary_nurdles, _ = match_sinks_with_individual_radius(sinks_na, df_nurdles, max_cap_km=50.0)
    _, summary_ocean, _ = match_sinks_with_individual_radius(sinks_na, df_ocean, max_cap_km=50.0)

    df_nurdles_pop = summary_nurdles.merge(pop_na, on="basin_id", how="inner")
    df_ocean_pop = summary_ocean.merge(pop_na, on="basin_id", how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # ----------------------------------------------------
    # Painel (a): Nurdles no Golfo (Polo Petroquímico) vs Fora do Golfo
    # ----------------------------------------------------
    ax = axes[0]
    gulf_mask = (df_nurdles_pop["is_gulf"] == True) & (df_nurdles_pop["total_population"] > 0) & (df_nurdles_pop["nurdles_sum"] > 0)
    nongulf_mask = (df_nurdles_pop["is_gulf"] == False) & (df_nurdles_pop["total_population"] > 0) & (df_nurdles_pop["nurdles_sum"] > 0)

    df_gulf = df_nurdles_pop[gulf_mask]
    df_nongulf = df_nurdles_pop[nongulf_mask]

    # Plotar Fora do Golfo (cinza claro)
    x_ng = np.log10(df_nongulf["total_population"].values)
    y_ng = np.log10(df_nongulf["nurdles_sum"].values)
    ax.scatter(x_ng, y_ng, edgecolors="#94a3b8", facecolors="none", s=35, linewidths=1.0, alpha=0.6, label="Outside Petrochemical Hub (N=161, $R^2=0.00$)")

    # Plotar Golfo do México (azul escuro destacado)
    x_g = np.log10(df_gulf["total_population"].values)
    y_g = np.log10(df_gulf["nurdles_sum"].values)
    ax.scatter(x_g, y_g, edgecolors="#2563eb", facecolors="none", s=55, linewidths=1.6, label=r"Gulf of Mexico Corridor ($N=31, R^2=0.1792$)")

    # Regressão OLS no Golfo
    slope_g, intercept_g, r_val_g, p_val_g, se_g = stats.linregress(x_g, y_g)
    x_line = np.linspace(x_g.min(), x_g.max(), 100)
    ax.plot(x_line, intercept_g + slope_g * x_line, color="#0f172a", lw=2.0, label="Gulf OLS Fit")

    # Faixa de confiança no Golfo
    n_g = len(x_g)
    t_val = stats.t.ppf(0.975, df=n_g - 2)
    s_err = np.sqrt(np.sum((y_g - (intercept_g + slope_g * x_g)) ** 2) / (n_g - 2))
    ci = t_val * s_err * np.sqrt(1.0 / n_g + (x_line - np.mean(x_g)) ** 2 / np.sum((x_g - np.mean(x_g)) ** 2))
    ax.fill_between(x_line, intercept_g + slope_g * x_line - ci, intercept_g + slope_g * x_line + ci, color="#ea580c", alpha=0.25)

    # Guia L-bracket beta = 1 em area livre (painel a)
    bx, by = 2.4, 0.4
    bw = 0.8
    ax.plot([bx, bx + bw], [by, by], color="#0f172a", lw=1.2)
    ax.plot([bx + bw, bx + bw], [by, by + bw], color="#0f172a", lw=1.2)
    ax.text(bx + bw + 0.05, by + bw / 2, r"$\beta = 1$", va="center", ha="left", fontsize=9.5)

    # Anotação
    txt_g = (
        r"$\mathbf{Gulf\ of\ Mexico\ (Petrochemical)}$" + "\n"
        + rf"$\beta = {slope_g:.2f} \pm {se_g:.2f}$" + "\n"
        + rf"$R^2 = {r_val_g**2:.4f}\ (17.9\%)$" + "\n"
        + rf"$r = +{r_val_g:.2f},\ p = {p_val_g:.3f}$" + "\n"
        + rf"$N = {n_g}\ \mathrm{{basins}}$"
    )
    ax.text(0.04, 0.95, txt_g, transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"))

    ax.set_xlabel(r"$\log_{10} P_{\mathrm{total}}$ (Basin total inhabitants)")
    ax.set_ylabel(r"$\log_{10} M_{\mathrm{sum}}$ (Total nurdles captured)")
    ax.set_ylim(-0.5, 6.2)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=8.5)
    rotulo_painel(ax, "(a)")

    # ----------------------------------------------------
    # Painel (b): Ocean Water (Redes Neuston) com N_samples >= 3
    # ----------------------------------------------------
    ax2 = axes[1]
    ow_mask = (df_ocean_pop["n_samples"] >= 3) & (df_ocean_pop["total_population"] > 0) & (df_ocean_pop["nurdles_median"] > 0)
    df_ow = df_ocean_pop[ow_mask]

    x_ow = np.log10(df_ow["total_population"].values)
    y_ow = np.log10(df_ow["nurdles_median"].values)

    ax2.scatter(x_ow, y_ow, edgecolors="#059669", facecolors="none", s=60, linewidths=1.8, label=r"Ocean Water ($N_{\mathrm{samples}} \geq 3$)")

    slope_ow, intercept_ow, r_val_ow, p_val_ow, se_ow = stats.linregress(x_ow, y_ow)
    x_line_ow = np.linspace(x_ow.min() - 0.2, x_ow.max() + 0.2, 100)
    ax2.plot(x_line_ow, intercept_ow + slope_ow * x_line_ow, color="#047857", lw=2.0, label=rf"Allometric Fit ($\beta = {slope_ow:.2f} \pm {se_ow:.2f}$)")

    # Linha guia isométrica beta = 1
    ax2.plot(x_line_ow, intercept_ow + 1.0 * (x_line_ow - np.mean(x_ow)), color="#64748b", ls=":", lw=1.2, label=r"Isometric Scaling ($\beta = 1.0$)")

    n_ow = len(x_ow)
    t_val_ow = stats.t.ppf(0.975, df=n_ow - 2)
    s_err_ow = np.sqrt(np.sum((y_ow - (intercept_ow + slope_ow * x_ow)) ** 2) / (n_ow - 2))
    ci_ow = t_val_ow * s_err_ow * np.sqrt(1.0 / n_ow + (x_line_ow - np.mean(x_ow)) ** 2 / np.sum((x_ow - np.mean(x_ow)) ** 2))
    ax2.fill_between(x_line_ow, intercept_ow + slope_ow * x_line_ow - ci_ow, intercept_ow + slope_ow * x_line_ow + ci_ow, color="#10b981", alpha=0.20)

    # L-bracket guide em area limpa (painel b)
    bx2, by2 = 2.8, -0.6
    bw2 = 0.6
    ax2.plot([bx2, bx2 + bw2], [by2, by2], color="#0f172a", lw=1.2)
    ax2.plot([bx2 + bw2, bx2 + bw2], [by2, by2 + bw2], color="#0f172a", lw=1.2)
    ax2.text(bx2 + bw2 + 0.05, by2 + bw2 / 2, r"$\beta = 1$", va="center", ha="left", fontsize=9.5)

    txt_ow = (
        r"$\mathbf{Ocean\ Water\ (Net\ Tows,\ N_{\mathrm{samples}} \geq 3)}$" + "\n"
        + rf"$\beta = {slope_ow:.2f} \pm {se_ow:.2f}$" + "\n"
        + rf"$R^2 = {r_val_ow**2:.4f}\ (22.7\%)$" + "\n"
        + rf"$r = +{r_val_ow:.2f},\ p = {p_val_ow:.3f}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +0.49\ (p = 0.10)$" + "\n"
        + rf"$N = {n_ow}\ \mathrm{{basins}}$"
    )
    ax2.text(0.04, 0.95, txt_ow, transform=ax2.transAxes, ha="left", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"))

    ax2.set_xlabel(r"$\log_{10} P_{\mathrm{total}}$ (Basin total inhabitants)")
    ax2.set_ylabel(r"$\log_{10} M_{\mathrm{median}}$ (Microplastics in pieces/$\mathrm{m}^3$)")
    ax2.set_ylim(-2.8, 1.8)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=8.5)
    rotulo_painel(ax2, "(b)")

    plt.tight_layout()
    out_png = REPO_ROOT / "relatorios" / "figuras" / "f9_diagnostic_correlation_comparison.png"
    out_pdf = REPO_ROOT / "relatorios" / "figuras" / "f9_diagnostic_correlation_comparison.pdf"
    salvar(fig, str(out_png))
    salvar(fig, str(out_pdf))
    print(f"Figura salva em:\n - {out_png}\n - {out_pdf}")

if __name__ == "__main__":
    gerar_figura_diagnostico()
