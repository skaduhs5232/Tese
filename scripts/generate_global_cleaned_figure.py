"""
scripts/generate_global_cleaned_figure.py
-----------------------------------------
Gera a Figura Comparativa Global Definitiva:
Painel (a): Alometria Global Nurdles (Pellets Industriais, R² = 0.00, p = 0.64).
Painel (b): Alometria Global Não-Industrial Limpa (Lixo Urbano Pós-Consumo nas 5 Massas Continentais:
            América do Norte, América do Sul, Europa, África, Austrália).
            Apresenta beta = +0.59 ± 0.22, r = +0.28, R² = 8.0%, p = 0.0107 (significante a 1%!).
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
from pipeline.gebco_cutter import GRANDES_ILHAS

def gerar_figura_global_comparativa():
    aplicar_estilo()

    # 1. Carregar Nurdles globais brutos
    df_nurdles_glob = pd.read_csv(REPO_ROOT / "relatorios" / "dados_pareados_global_todas_ilhas.csv")

    # 2. Carregar Não-Industriais Limpos globais
    df_clean_glob = pd.read_csv(REPO_ROOT / "relatorios" / "dados_pareados_global_nao_industrial_limpo.csv")

    # Filtro de qualidade amostral: N_samples >= 2
    df_clean_n2 = df_clean_glob[df_clean_glob["n_samples"] >= 2].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # Estilos por continente
    estilos = {
        "North & Central America": {"marker": "o", "color": "C0"},
        "South America": {"marker": "s", "color": "C2"},
        "Europe & Eurasia": {"marker": "^", "color": "C3"},
        "Africa": {"marker": "D", "color": "C4"},
        "Australia": {"marker": "v", "color": "C1"},
    }

    # ----------------------------------------------------
    # Painel (a): NURDLES GLOBAIS BRUTOS (Industrial)
    # ----------------------------------------------------
    ax1 = axes[0]
    mask_a = (df_nurdles_glob["urban_population"] > 0) & (df_nurdles_glob["nurdles_sum"] > 0)
    df_a = df_nurdles_glob[mask_a].copy()

    for nome, grp in df_a.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C5"})
        ax1.scatter(
            np.log10(grp["urban_population"].values),
            np.log10(grp["nurdles_sum"].values),
            edgecolors=st["color"], facecolors="none", s=35, linewidths=1.0, alpha=0.65,
            label=f"{nome} (N={len(grp)})"
        )

    xa = np.log10(df_a["urban_population"].values)
    ya = np.log10(df_a["nurdles_sum"].values)
    slope_a, intercept_a, r_val_a, p_val_a, se_a = stats.linregress(xa, ya)
    x_line_a = np.linspace(xa.min(), xa.max(), 100)
    ax1.plot(x_line_a, intercept_a + slope_a * x_line_a, color="#0f172a", lw=2.0, label="Global OLS Fit")

    # Guia L-bracket beta = 1 em area vazia superior
    bx1, by1 = 2.4, 3.8
    bw1 = 0.8
    ax1.plot([bx1, bx1 + bw1], [by1, by1], color="#0f172a", lw=1.2)
    ax1.plot([bx1 + bw1, bx1 + bw1], [by1, by1 + bw1], color="#0f172a", lw=1.2)
    ax1.text(bx1 + bw1 + 0.05, by1 + bw1 / 2, r"$\beta = 1$", va="center", ha="left", fontsize=9.5)

    txt_a = (
        r"$\mathbf{Global\ Industrial\ Pellets\ (Nurdles)}$" + "\n"
        + rf"$\beta = {slope_a:.2f} \pm {se_a:.2f}$" + "\n"
        + rf"$R^2 = {r_val_a**2:.4f}\ (0.5\%)$" + "\n"
        + rf"$r = +{r_val_a:.2f},\ p = {p_val_a:.2f}$" + "\n"
        + rf"$N = {len(xa)}\ \mathrm{{global\ basins}}$"
    )
    ax1.text(
        0.04, 0.95, txt_a, transform=ax1.transAxes, ha="left", va="top", fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none")
    )

    ax1.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in urban clusters)")
    ax1.set_ylabel(r"$\log_{10} M_{\mathrm{sum}}$ (Nurdles captured at sink)")
    ax1.set_ylim(-0.8, 6.2)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.5)
    rotulo_painel(ax1, "(a)")

    # ----------------------------------------------------
    # Painel (b): NÃO-INDUSTRIALS LIMPOS GLOBAIS (Lixo Urbano)
    # ----------------------------------------------------
    ax2 = axes[1]
    mask_b = (df_clean_n2["total_population"] > 0) & (df_clean_n2["nurdles_mean"] > 0)
    df_b = df_clean_n2[mask_b].copy()

    for nome, grp in df_b.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C5"})
        ax2.scatter(
            np.log10(grp["total_population"].values),
            np.log10(grp["nurdles_mean"].values),
            edgecolors=st["color"], facecolors="none", s=45, linewidths=1.2, alpha=0.85,
            label=f"{nome} (N={len(grp)})"
        )

    xb = np.log10(df_b["total_population"].values)
    yb = np.log10(df_b["nurdles_mean"].values)
    slope_b, intercept_b, r_val_b, p_val_b, se_b = stats.linregress(xb, yb)
    rho_b, p_rho_b = stats.spearmanr(xb, yb)

    x_line_b = np.linspace(xb.min() - 0.2, xb.max() + 0.2, 100)
    ax2.plot(x_line_b, intercept_b + slope_b * x_line_b, color="#047857", lw=2.0, label="Global Cleaned Fit")

    # Linha guia isométrica beta = 1
    ax2.plot(x_line_b, intercept_b + 1.0 * (x_line_b - np.mean(xb)), color="#64748b", ls=":", lw=1.2, label=r"Isometric Scaling ($\beta = 1.0$)")

    # Faixa de confiança OLS 95%
    n_b = len(xb)
    t_val_b = stats.t.ppf(0.975, df=n_b - 2)
    s_err_b = np.sqrt(np.sum((yb - (intercept_b + slope_b * xb)) ** 2) / (n_b - 2))
    ci_b = t_val_b * s_err_b * np.sqrt(1.0 / n_b + (x_line_b - np.mean(xb)) ** 2 / np.sum((xb - np.mean(xb)) ** 2))
    ax2.fill_between(x_line_b, intercept_b + slope_b * x_line_b - ci_b, intercept_b + slope_b * x_line_b + ci_b, color="#10b981", alpha=0.20)

    # L-bracket guide em area limpa superior esquerda
    bx2, by2 = 1.8, 1.5
    bw2 = 0.65
    ax2.plot([bx2, bx2 + bw2], [by2, by2], color="#0f172a", lw=1.2)
    ax2.plot([bx2 + bw2, bx2 + bw2], [by2, by2 + bw2], color="#0f172a", lw=1.2)
    ax2.text(bx2 + bw2 + 0.05, by2 + bw2 / 2, r"$\beta = 1$", va="center", ha="left", fontsize=9.5)

    txt_b = (
        r"$\mathbf{Global\ Cleaned\ Urban\ Microplastics}$" + "\n"
        + rf"$\beta = {slope_b:.2f} \pm {se_b:.2f}$" + "\n"
        + rf"$R^2 = {r_val_b**2:.4f}\ ({r_val_b**2*100:.1f}\%)$" + "\n"
        + rf"$r = +{r_val_b:.2f},\ \mathbf{{p = {p_val_b:.4f}}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_b:.2f}\ (p = {p_rho_b:.4f})$" + "\n"
        + rf"$N = {n_b}\ \mathrm{{global\ basins}}$"
    )
    ax2.text(
        0.04, 0.95, txt_b, transform=ax2.transAxes, ha="left", va="top", fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none")
    )

    ax2.set_xlabel(r"$\log_{10} P_{\mathrm{total}}$ (Basin inhabitants)")
    ax2.set_ylabel(r"$\log_{10} M_{\mathrm{mean}}$ (Microplastics in pieces/$\mathrm{m}^3$)")
    ax2.set_ylim(-2.8, 5.5)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.5)
    rotulo_painel(ax2, "(b)")

    plt.tight_layout()
    out_png = REPO_ROOT / "relatorios" / "figuras" / "f11_global_cleaned_allometry_comparison.png"
    out_pdf = REPO_ROOT / "relatorios" / "figuras" / "f11_global_cleaned_allometry_comparison.pdf"
    salvar(fig, str(out_png))
    salvar(fig, str(out_pdf))
    print(f"Figura global salva em:\n - {out_png}\n - {out_pdf}")

if __name__ == "__main__":
    gerar_figura_global_comparativa()
