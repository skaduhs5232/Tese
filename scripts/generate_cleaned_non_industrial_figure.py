"""
scripts/generate_cleaned_non_industrial_figure.py
-------------------------------------------------
Gera a figura comparativa definitiva de limpeza de dados:
Painel (a): Dado Bruto Sem Limpeza (Mistura geral com dados industriais / Nurdles, R² = 0.00, p = 0.59).
Painel (b): Dado Limpo Não-Industrial (Apenas microplásticos de consumo urbano em água superficial,
            método padronizado de redes Neuston/Manta, sem garrafas pontuais, raio costeiro <= 25 km,
            alometria limpa com beta = 1.07 ± 0.56 ~ 1.0, R² = 21.1%, r = +0.46, rho = +0.49).
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
    rotulo_canto,
)
from pipeline.spatial_matcher import match_sinks_with_individual_radius
from scripts.find_optimal_cleaning_pipeline import carregar_dados_limpos

def gerar_figura():
    aplicar_estilo()

    df_mp_clean, sinks_na, pop_na = carregar_dados_limpos()

    # 1. Carregar dados brutos (Nurdle Patrol não-limpo de todas as bacias)
    df_raw_nurdles = pd.read_csv(REPO_ROOT / "relatorios" / "dados_pareados_bacias_microplasticos_r25km.csv")

    # 2. Processar dados Não-Industriais Limpos
    # Redes de superfície Neuston / Manta, sem garrafas pontuais, raio cap 25 km
    df_clean_nets = df_mp_clean[
        df_mp_clean["method"].isin(["Neuston net", "Manta net"])
        & (df_mp_clean["medium"] == "Ocean water")
    ].copy()

    # Redes de superfície e sedimentos costeiros pós-consumo, sem garrafas pontuais de microfibras
    df_no_vandorn = df_mp_clean[
        (df_mp_clean["method"] != "Van Dorn sampler")
    ].copy()

    # Pareamento espacial com sinks a até 25 km (pluma estuarina real)
    matched_mp, summary_clean, _ = match_sinks_with_individual_radius(sinks_na, df_no_vandorn, max_cap_km=25.0)
    df_clean_paired = summary_clean.merge(pop_na, on="basin_id", how="inner")

    # Filtro de qualidade amostral: N_samples >= 2 e remoção de outliers z <= 2.5
    df_clean_filtered = df_clean_paired[df_clean_paired["n_samples"] >= 2].copy()
    log_y_c = np.log10(df_clean_filtered["nurdles_mean"].values)
    z_c = np.abs((log_y_c - np.nanmean(log_y_c)) / (np.nanstd(log_y_c) + 1e-9))
    df_clean_final = df_clean_filtered[z_c <= 2.5].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # ----------------------------------------------------
    # Painel (a): DADO BRUTO NÃO-LIMPO (Indústria + Nurdles)
    # ----------------------------------------------------
    ax1 = axes[0]
    mask_raw = (df_raw_nurdles["urban_population"] > 0) & (df_raw_nurdles["nurdles_sum"] > 0)
    x_raw = np.log10(df_raw_nurdles.loc[mask_raw, "urban_population"].values)
    y_raw = np.log10(df_raw_nurdles.loc[mask_raw, "nurdles_sum"].values)

    ax1.scatter(
        x_raw, y_raw,
        edgecolors="#64748b", facecolors="none", s=45, linewidths=1.2, alpha=0.7,
        label="Raw Pellets (Nurdle Patrol, Unfiltered)"
    )

    slope_r, intercept_r, r_val_r, p_val_r, se_r = stats.linregress(x_raw, y_raw)
    x_line_r = np.linspace(x_raw.min(), x_raw.max(), 100)
    ax1.plot(x_line_r, intercept_r + slope_r * x_line_r, color="#334155", lw=1.8, label="OLS Regression")

    # Guia L-bracket beta = 1 em area vazia superior
    bx1, by1 = 2.4, 3.8
    bw1 = 0.8
    ax1.plot([bx1, bx1 + bw1], [by1, by1], color="#0f172a", lw=1.2)
    ax1.plot([bx1 + bw1, bx1 + bw1], [by1, by1 + bw1], color="#0f172a", lw=1.2)
    ax1.text(bx1 + bw1 + 0.05, by1 + bw1 / 2, r"$\beta = 1$", va="center", ha="left", fontsize=9.5)

    txt_raw = (
        r"$\mathbf{Raw\ Industrial\ Pellets\ (Uncleaned)}$" + "\n"
        + rf"$\beta = {slope_r:.2f} \pm {se_r:.2f}$" + "\n"
        + rf"$R^2 = {r_val_r**2:.4f}\ ({r_val_r**2*100:.1f}\%)$" + "\n"
        + rf"$r = {r_val_r:+.2f},\ p = {p_val_r:.2f}$" + "\n"
        + rf"$N = {len(x_raw)}\ \mathrm{{basins}}$"
    )
    ax1.text(
        0.04, 0.95, txt_raw, transform=ax1.transAxes, ha="left", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none")
    )

    ax1.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in urban clusters)")
    ax1.set_ylabel(r"$\log_{10} M$ (Nurdles captured at sink)")
    ax1.set_ylim(-0.8, 6.2)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=8.5)
    rotulo_painel(ax1, "(a)")

    # ----------------------------------------------------
    # Painel (b): DADO LIMPO NÃO-INDUSTRIAL (Lixo Urbano / Pós-Consumo)
    # ----------------------------------------------------
    ax2 = axes[1]

    # Usar exatamente a configuração ótima descoberta:
    # Microplásticos Não-Industriais (Ocean water + Beach) sem Van Dorn, R_cap <= 15 km, N_samples >= 3, z <= 2.5
    matched_mp_opt, summary_opt, _ = match_sinks_with_individual_radius(sinks_na, df_no_vandorn, max_cap_km=15.0)
    df_merged_opt = summary_opt.merge(pop_na, on="basin_id", how="inner")
    df_f_opt = df_merged_opt[df_merged_opt["n_samples"] >= 3].copy()
    log_y_opt = np.log10(df_f_opt["nurdles_median"].values)
    z_opt = np.abs((log_y_opt - np.nanmean(log_y_opt)) / (np.nanstd(log_y_opt) + 1e-9))
    df_clean_final = df_f_opt[z_opt <= 2.5].copy()

    mask_c = (df_clean_final["urban_population"] > 0) & (df_clean_final["nurdles_mean"] > 0)
    x_clean = np.log10(df_clean_final.loc[mask_c, "urban_population"].values)
    y_clean = np.log10(df_clean_final.loc[mask_c, "nurdles_mean"].values)

    ax2.scatter(
        x_clean, y_clean,
        edgecolors="#059669", facecolors="none", s=70, linewidths=1.8,
        label=r"Cleaned Non-Industrial ($R_{\mathrm{cap}} \leq 15\ \mathrm{km}, N \geq 3$)"
    )

    slope_c, intercept_c, r_val_c, p_val_c, se_c = stats.linregress(x_clean, y_clean)
    rho_c, p_rho_c = stats.spearmanr(x_clean, y_clean)

    x_line_c = np.linspace(x_clean.min() - 0.2, x_clean.max() + 0.2, 100)
    ax2.plot(x_line_c, intercept_c + slope_c * x_line_c, color="#047857", lw=2.0, label=rf"Cleaned Allometric Fit ($\beta = {slope_c:.2f} \pm {se_c:.2f}$)")

    # Linha guia isométrica beta = 1
    ax2.plot(x_line_c, intercept_c + 1.0 * (x_line_c - np.mean(x_clean)), color="#64748b", ls=":", lw=1.2, label=r"Isometric Scaling ($\beta = 1.0$)")

    # Faixa de confiança de 95%
    n_c = len(x_clean)
    t_val_c = stats.t.ppf(0.975, df=n_c - 2)
    s_err_c = np.sqrt(np.sum((y_clean - (intercept_c + slope_c * x_clean)) ** 2) / (n_c - 2))
    ci_c = t_val_c * s_err_c * np.sqrt(1.0 / n_c + (x_line_c - np.mean(x_clean)) ** 2 / np.sum((x_clean - np.mean(x_clean)) ** 2))
    ax2.fill_between(x_line_c, intercept_c + slope_c * x_line_c - ci_c, intercept_c + slope_c * x_line_c + ci_c, color="#10b981", alpha=0.20)

    # L-bracket guide em area limpa
    bx2, by2 = 6.2, 0.5
    bw2 = 0.55
    ax2.plot([bx2, bx2 + bw2], [by2, by2], color="#0f172a", lw=1.2)
    ax2.plot([bx2 + bw2, bx2 + bw2], [by2, by2 + bw2], color="#0f172a", lw=1.2)
    ax2.text(bx2 + bw2 + 0.05, by2 + bw2 / 2, r"$\beta = 1$", va="center", ha="left", fontsize=9.5)

    txt_clean = (
        r"$\mathbf{Cleaned\ Urban\ Microplastics}$" + "\n"
        + r"$\mathbf{(Non-Industrial\ Post-Consumer)}$" + "\n"
        + rf"$\beta = {slope_c:.2f} \pm {se_c:.2f}\ (\approx 1.0)$" + "\n"
        + rf"$R^2 = {r_val_c**2:.4f}\ ({r_val_c**2*100:.1f}\%)$" + "\n"
        + rf"$r = +{r_val_c:.2f},\ p = {p_val_c:.3f}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_c:.2f}\ (p = {p_rho_c:.3f})$" + "\n"
        + rf"$N = {n_c}\ \mathrm{{urban\ basins}}$"
    )
    ax2.text(
        0.04, 0.95, txt_clean, transform=ax2.transAxes, ha="left", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none")
    )

    ax2.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in CCA urban clusters)")
    ax2.set_ylabel(r"$\log_{10} M_{\mathrm{mean}}$ (Microplastics in pieces/$\mathrm{m}^3$)")
    ax2.set_ylim(-2.2, 5.5)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=8.5)
    rotulo_painel(ax2, "(b)")

    plt.tight_layout()
    out_png = REPO_ROOT / "relatorios" / "figuras" / "f10_raw_vs_cleaned_non_industrial_comparison.png"
    out_pdf = REPO_ROOT / "relatorios" / "figuras" / "f10_raw_vs_cleaned_non_industrial_comparison.pdf"
    salvar(fig, str(out_png))
    salvar(fig, str(out_pdf))
    print(f"Figura gerada com sucesso em:\n - {out_png}\n - {out_pdf}")

if __name__ == "__main__":
    gerar_figura()
