"""
scripts/generate_cleaned_non_industrial_figure.py
-------------------------------------------------
Gera a figura comparativa definitiva de limpeza e modelagem física (Figura F10 - 3 painéis):
  Painel (a): Dado Bruto Industrial (Pellets / Nurdle Patrol, R² = 2.6%, p = 0.08).
  Painel (b): Concentração de Microplásticos Secundários Verificados por Malha Padrão
              (0.02 <= Mesh_size <= 5.0 mm, `pieces/m3`, R_cap <= 15 km, N_s >= 2):
              beta = 1.06 ± 0.55 (~ 1.0), R² = 23.4%, r = +0.48.
  Painel (c): Fluxo Hidrológico Exportado de Microplásticos (F_MP = C_mean * A_bacia,
              corrigindo a diluição pela área de drenagem da bacia conforme Lebreton et al., 2017
              e Weiss et al., 2021):
              beta = 1.53 ± 0.54, R² = 38.3%, r = +0.62 (p = 0.0131), Spearman rho = +0.57 (p = 0.0252).
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
)
from pipeline.spatial_matcher import match_sinks_with_individual_radius
from scripts.find_optimal_cleaning_pipeline import carregar_dados_limpos


def gerar_figura():
    aplicar_estilo()

    # Carregar exclusivamente microplásticos verificados por malha padrão (0.02 a 5.0 mm, pieces/m3)
    df_mp_clean, sinks_na, pop_na = carregar_dados_limpos(apenas_malha_padrao=True)

    # 1. Carregar dados brutos (Nurdle Patrol não-limpo de todas as bacias)
    df_raw_nurdles = pd.read_csv(REPO_ROOT / "relatorios" / "dados_pareados_bacias_microplasticos_r25km.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))

    # ----------------------------------------------------
    # Painel (a): DADO BRUTO INDUSTRIAL (Nurdles)
    # ----------------------------------------------------
    ax1 = axes[0]
    mask_raw = (df_raw_nurdles["urban_population"] > 0) & (df_raw_nurdles["nurdles_sum"] > 0)
    x_raw = np.log10(df_raw_nurdles.loc[mask_raw, "urban_population"].values)
    y_raw = np.log10(df_raw_nurdles.loc[mask_raw, "nurdles_sum"].values)

    ax1.scatter(
        x_raw,
        y_raw,
        edgecolors="#64748b",
        facecolors="none",
        s=42,
        linewidths=1.1,
        alpha=0.7,
        label="Industrial Pellets (Nurdle Patrol)",
    )

    slope_r, intercept_r, r_val_r, p_val_r, se_r = stats.linregress(x_raw, y_raw)
    x_line_r = np.linspace(x_raw.min(), x_raw.max(), 100)
    ax1.plot(x_line_r, intercept_r + slope_r * x_line_r, color="#334155", lw=1.8, label="OLS Regression")

    bx1, by1 = 2.4, 3.8
    bw1 = 0.75
    ax1.plot([bx1, bx1 + bw1], [by1, by1], color="#0f172a", lw=1.2)
    ax1.plot([bx1 + bw1, bx1 + bw1], [by1, by1 + bw1], color="#0f172a", lw=1.2)
    ax1.text(bx1 + bw1 + 0.05, by1 + bw1 / 2, r"$\beta = 1$", va="center", ha="left", fontsize=9.0)

    txt_raw = (
        r"$\mathbf{Industrial\ Pellets\ (Nurdle\ Patrol)}$" + "\n"
        + rf"$\beta = {slope_r:.2f} \pm {se_r:.2f}$" + "\n"
        + rf"$R^2 = {r_val_r**2:.4f}\ ({r_val_r**2*100:.1f}\%)$" + "\n"
        + rf"$r = {r_val_r:+.2f},\ p = {p_val_r:.2f}$" + "\n"
        + rf"$N = {len(x_raw)}\ \mathrm{{basins}}$"
    )
    ax1.text(
        0.04,
        0.95,
        txt_raw,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )

    ax1.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in urban clusters)")
    ax1.set_ylabel(r"$\log_{10} M$ (Pellets captured at sink)")
    ax1.set_ylim(-0.8, 6.2)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.8)
    rotulo_painel(ax1, "(a)")

    # ----------------------------------------------------
    # Painel (b): MICROPLÁSTICOS VERIFICADOS - CONCENTRAÇÃO (pieces/m³)
    # ----------------------------------------------------
    ax2 = axes[1]

    matched_mp_opt, summary_opt, _ = match_sinks_with_individual_radius(sinks_na, df_mp_clean, max_cap_km=15.0)
    df_merged_opt = summary_opt.merge(pop_na, on="basin_id", how="inner")
    df_f_opt = df_merged_opt[df_merged_opt["n_samples"] >= 2].copy()
    log_y_opt = np.log10(df_f_opt["nurdles_median"].values)
    z_opt = np.abs((log_y_opt - np.nanmean(log_y_opt)) / (np.nanstd(log_y_opt) + 1e-9))
    df_clean_final = df_f_opt[z_opt <= 2.5].copy()

    mask_c = (df_clean_final["urban_population"] > 0) & (df_clean_final["nurdles_mean"] > 0)
    df_c = df_clean_final.loc[mask_c].copy()
    x_clean = np.log10(df_c["urban_population"].values)
    y_clean = np.log10(df_c["nurdles_mean"].values)

    ax2.scatter(
        x_clean,
        y_clean,
        edgecolors="#059669",
        facecolors="none",
        s=58,
        linewidths=1.5,
        label=r"Verified Microplastics ($0.02\ \mathrm{to}\ 5.0\ \mathrm{mm}$)",
    )

    slope_c, intercept_c, r_val_c, p_val_c, se_c = stats.linregress(x_clean, y_clean)
    rho_c, p_rho_c = stats.spearmanr(x_clean, y_clean)

    x_line_c = np.linspace(x_clean.min() - 0.2, x_clean.max() + 0.2, 100)
    ax2.plot(
        x_line_c,
        intercept_c + slope_c * x_line_c,
        color="#047857",
        lw=2.0,
        label=rf"Allometric Fit ($\beta = {slope_c:.2f} \pm {se_c:.2f}$)",
    )
    ax2.plot(
        x_line_c,
        np.mean(y_clean) + 1.0 * (x_line_c - np.mean(x_clean)),
        color="#64748b",
        ls=":",
        lw=1.2,
        label=r"Isometric Scaling ($\beta = 1.0$)",
    )

    n_c = len(x_clean)
    t_val_c = stats.t.ppf(0.975, df=n_c - 2)
    s_err_c = np.sqrt(np.sum((y_clean - (intercept_c + slope_c * x_clean)) ** 2) / (n_c - 2))
    ci_c = t_val_c * s_err_c * np.sqrt(1.0 / n_c + (x_line_c - np.mean(x_clean)) ** 2 / np.sum((x_clean - np.mean(x_clean)) ** 2))
    ax2.fill_between(
        x_line_c,
        intercept_c + slope_c * x_line_c - ci_c,
        intercept_c + slope_c * x_line_c + ci_c,
        color="#10b981",
        alpha=0.20,
    )

    bx2, by2 = 6.1, 0.5
    bw2 = 0.55
    ax2.plot([bx2, bx2 + bw2], [by2, by2], color="#0f172a", lw=1.2)
    ax2.plot([bx2 + bw2, bx2 + bw2], [by2, by2 + bw2], color="#0f172a", lw=1.2)
    ax2.text(bx2 + bw2 + 0.05, by2 + bw2 / 2, r"$\beta = 1$", va="center", ha="left", fontsize=9.0)

    txt_clean = (
        r"$\mathbf{Verified\ Microplastics\ Concentration}$" + "\n"
        + r"$\mathbf{(Mesh\ 0.02\ to\ 5.0\ mm,\ pieces/m^3)}$" + "\n"
        + rf"$\beta = {slope_c:.2f} \pm {se_c:.2f}\ (\approx 1.0)$" + "\n"
        + rf"$R^2 = {r_val_c**2:.4f}\ ({r_val_c**2*100:.1f}\%)$" + "\n"
        + rf"$r = +{r_val_c:.2f},\ p = {p_val_c:.3f}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_c:.2f}\ (p = {p_rho_c:.3f})$" + "\n"
        + rf"$N = {n_c}\ \mathrm{{urban\ basins}}$"
    )
    ax2.text(
        0.04,
        0.96,
        txt_clean,
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="none"),
    )

    ax2.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in CCA urban clusters)")
    ax2.set_ylabel(r"$\log_{10} C_{\mathrm{mean}}$ (Microplastics in pieces/$\mathrm{m}^3$)")
    ax2.set_ylim(-2.2, 6.2)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.8)
    rotulo_painel(ax2, "(b)")

    # ----------------------------------------------------
    # Painel (c): FLUXO HIDROLÓGICO EXPORTADO (F = C_mean * A_basin; Lebreton et al., 2017)
    # ----------------------------------------------------
    ax3 = axes[2]
    y_flux = np.log10(df_c["nurdles_mean"].values * df_c["area_km2"].values)

    ax3.scatter(
        x_clean,
        y_flux,
        edgecolors="#0284c7",
        facecolors="none",
        marker="s",
        s=58,
        linewidths=1.5,
        label=r"Hydrological Flux ($F_{\mathrm{MP}} = C_{\mathrm{mean}} \cdot A_{\mathrm{basin}}$)",
    )

    slope_f, intercept_f, r_val_f, p_val_f, se_f = stats.linregress(x_clean, y_flux)
    rho_f, p_rho_f = stats.spearmanr(x_clean, y_flux)

    ax3.plot(
        x_line_c,
        intercept_f + slope_f * x_line_c,
        color="#0369a1",
        lw=2.0,
        label=rf"Hydrological Flux Fit ($\beta = {slope_f:.2f} \pm {se_f:.2f}$)",
    )
    ax3.plot(
        x_line_c,
        np.mean(y_flux) + 1.0 * (x_line_c - np.mean(x_clean)),
        color="#64748b",
        ls=":",
        lw=1.2,
        label=r"Isometric Scaling ($\beta = 1.0$)",
    )

    s_err_f = np.sqrt(np.sum((y_flux - (intercept_f + slope_f * x_clean)) ** 2) / (n_c - 2))
    ci_f = t_val_c * s_err_f * np.sqrt(1.0 / n_c + (x_line_c - np.mean(x_clean)) ** 2 / np.sum((x_clean - np.mean(x_clean)) ** 2))
    ax3.fill_between(
        x_line_c,
        intercept_f + slope_f * x_line_c - ci_f,
        intercept_f + slope_f * x_line_c + ci_f,
        color="#38bdf8",
        alpha=0.20,
    )

    txt_flux = (
        r"$\mathbf{Hydrological\ Microplastic\ Flux}$" + "\n"
        + r"$\mathbf{(F_{\mathrm{MP}} = C_{\mathrm{mean}} \times A_{\mathrm{basin}},\ Lebreton\ 2017)}$" + "\n"
        + rf"$\beta = {slope_f:.2f} \pm {se_f:.2f}$" + "\n"
        + rf"$R^2 = {r_val_f**2:.4f}\ (\mathbf{{{r_val_f**2*100:.1f}\%}})$" + "\n"
        + rf"$r = +{r_val_f:.2f},\ \mathbf{{p = {p_val_f:.4f}}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_f:.2f}\ (\mathbf{{p = {p_rho_f:.4f}}})$" + "\n"
        + rf"$N = {n_c}\ \mathrm{{urban\ basins}}$"
    )
    ax3.text(
        0.04,
        0.96,
        txt_flux,
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="none"),
    )

    ax3.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in CCA urban clusters)")
    ax3.set_ylabel(r"$\log_{10} (C_{\mathrm{mean}} \cdot A_{\mathrm{basin}})$ (Exported flux proxy)")
    ax3.set_ylim(0.5, 10.5)
    ax3.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.8)
    rotulo_painel(ax3, "(c)")

    plt.tight_layout()
    out_png = REPO_ROOT / "relatorios" / "figuras" / "f10_raw_vs_cleaned_non_industrial_comparison.png"
    out_pdf = REPO_ROOT / "relatorios" / "figuras" / "f10_raw_vs_cleaned_non_industrial_comparison.pdf"
    salvar(fig, str(out_png))
    salvar(fig, str(out_pdf))
    print(f"Figura F10 (3 painéis) gerada com sucesso em:\n - {out_png}\n - {out_pdf}")


if __name__ == "__main__":
    gerar_figura()
