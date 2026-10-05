"""
Gera os gráficos do artigo da Lei de Zipf em altíssima qualidade (vetorial e 300 DPI),
com proporções físicas calibradas para a largura do texto no LaTeX (\textwidth = ~17.4 cm),
com barra de legenda horizontal unificada abaixo dos painéis,
garantindo 100% de visibilidade de todos os dados sem nenhuma sobreposição.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Diretórios
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIGO_FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'texas_legado')

os.makedirs(ARTIGO_FIGS_DIR, exist_ok=True)

# Carrega os dados
with open(os.path.join(RESULTS_DIR, 'zipf_resumo.json'), 'r', encoding='utf-8') as f:
    resumo = json.load(f)

df_us = pd.read_csv(os.path.join(RESULTS_DIR, 'zipf_clusters_US.csv'))
df_tx = pd.read_csv(os.path.join(RESULTS_DIR, 'zipf_clusters_TX.csv'))

# Configuração tipográfica profissional - compatível com LaTeX Computer / Latin Modern
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Latin Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'font.size': 8.5,
    'axes.labelsize': 9.0,
    'axes.titlesize': 9.0,
    'xtick.labelsize': 8.0,
    'ytick.labelsize': 8.0,
    'legend.fontsize': 7.4,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.linestyle': ':',
    'axes.linewidth': 0.75,
    'lines.linewidth': 1.3,
    'mathtext.fontset': 'cm'
})

fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.55), dpi=300)

# -------------------------------------------------------------
# (a) EUA contíguo
# -------------------------------------------------------------
fit_us = resumo['US']['fit']
x_us = np.sort(df_us['pop'].to_numpy())[::-1]
r_us = np.arange(1, len(x_us) + 1)
nt_us = fit_us['n']
xmin_us = int(round(fit_us['xmin']))

ax = axes[0]
l1 = ax.loglog(r_us, x_us, 'o', ms=1.8, mfc='#e05a2b', mec='none', alpha=0.45, label='Todas as manchas')[0]
l2 = ax.loglog(r_us[:nt_us], x_us[:nt_us], 'o', ms=2.0, mfc='none', mec='#111111', mew=0.55, label='Cauda ajustada')[0]
rr_us = np.logspace(0, np.log10(nt_us), 60)
a_us = fit_us['alpha']
l3 = ax.loglog(rr_us, fit_us['xmin'] * (rr_us / nt_us) ** (-a_us), '-', c='#0b2545', lw=1.6, label='Ajuste MLE')[0]
l4 = ax.loglog(rr_us, fit_us['xmin'] * (rr_us / nt_us) ** (-1.0), '--', c='#666666', lw=1.2, label=r'Zipf ideal ($\alpha = 1$)')[0]

ax.set_ylim(8e1, 1e9)
ax.set_xlim(0.7, 1.8e4)
ax.set_xlabel(r'Posto $r$')
ax.set_ylabel(r'População $S(r)$')
ax.set_title(r'(a) EUA contíguo (8.401 manchas)' '\n' r'KS = 0.014 ($p = 0.74$)')

# -------------------------------------------------------------
# (b) Costa do Texas (recorte da dissertação)
# -------------------------------------------------------------
fit_tx = resumo['TX']['fit']
x_tx = np.sort(df_tx['pop'].to_numpy())[::-1]
r_tx = np.arange(1, len(x_tx) + 1)
nt_tx = fit_tx['n']
xmin_tx = int(round(fit_tx['xmin']))

ax = axes[1]
ax.loglog(r_tx, x_tx, 'o', ms=2.2, mfc='#2a75d3', mec='none', alpha=0.6)
ax.loglog(r_tx[:nt_tx], x_tx[:nt_tx], 'o', ms=2.4, mfc='none', mec='#111111', mew=0.55)
rr_tx = np.logspace(0, np.log10(nt_tx), 60)
a_tx = fit_tx['alpha']
ax.loglog(rr_tx, fit_tx['xmin'] * (rr_tx / nt_tx) ** (-a_tx), '-', c='#0b2545', lw=1.6)
ax.loglog(rr_tx, fit_tx['xmin'] * (rr_tx / nt_tx) ** (-1.0), '--', c='#666666', lw=1.2)

ax.set_ylim(4e1, 5e7)
ax.set_xlim(0.7, 500)
ax.set_xlabel(r'Posto $r$')
ax.set_ylabel(r'População $S(r)$')
ax.set_title(r'(b) Costa do Texas (dissertação)' '\n' r'KS = 0.071 ($p = 0.34$)')

# -------------------------------------------------------------
# (c) Sensibilidade ao parâmetro l
# -------------------------------------------------------------
ax = axes[2]
v_l = resumo['varredura_l']
ls = np.array(v_l['l_m']) / 1000
alphas = v_l['alpha']
erros = v_l['erro']

l5 = ax.errorbar(ls, alphas, yerr=erros, fmt='s-', ms=4.2, c='#0b2545', ecolor='#666666', capsize=2.5, lw=1.3,
                 label=r'$\alpha(\ell)$ ajustado')
ax.axhline(1.0, ls='--', c='#666666', lw=1.2)
ax.set_xscale('log')
ax.set_xticks(ls)
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax.set_xlabel(r'$\ell$ (km)')
ax.set_ylabel(r'Expoente $\alpha$')
ax.set_ylim(0.92, 1.58)
ax.set_title(r'(c) Sensibilidade a $\ell$ do CCA' '\n' r'EUA contíguo ($x_{\min}$ fixo)')

# Rótulos diretos junto às curvas para clareza imediata
ax.text(1.05, 1.02, r'Zipf ($\alpha=1$)', color='#666666', fontsize=7.2, va='bottom')
ax.text(1.05, 1.34, r'$\alpha(\ell)$', color='#0b2545', fontsize=8.0, fontweight='bold', va='bottom')

# Legenda unificada na parte inferior
handles = [l1, l2, l3, l4, l5]
labels = ['Todas as manchas', 'Cauda ajustada', 'Ajuste MLE', r'Zipf ideal ($\alpha = 1$)', r'$\alpha(\ell)$ ajustado']
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=5, frameon=True, framealpha=0.95, edgecolor='#cccccc',
           fontsize=7.3, handletextpad=0.25, columnspacing=1.1)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

# Exporta em PDF vetorial e PNG de alta resolução
pdf_path = os.path.join(ARTIGO_FIGS_DIR, 'zipf.pdf')
png_path = os.path.join(ARTIGO_FIGS_DIR, 'zipf.png')

plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"Figuras atualizadas e salvas em:\n - {pdf_path}\n - {png_path}")
