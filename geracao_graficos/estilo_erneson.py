"""
geracao_graficos/estilo_erneson.py
----------------------------------
Módulo oficial de estilo visual e primitivas de plotagem científica
fundamentado nas publicações do Prof. Dr. Erneson A. Oliveira:
  - Oliveira et al. (2019), Sci. Rep. 9:9845 (IPBA)
  - Oliveira et al. (2018), R. Soc. Open Sci. 5:180468 (CCA e Zipf)
  - Oliveira et al. (2014), Sci. Rep. 4:4235 (Alometria e CCA)
  - Operti, Oliveira et al. (2017), arXiv:1706.05139 (Calibração D* x L)
E nas regras específicas da dissertação estabelecidas em `diretrizes/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
WORLD_GEOJSON = ROOT_DIR / "diretrizes" / "assets" / "world_land_110m.geojson"

FIGSIZE_MAPA_MUNDI = (7.0, 3.6)
FIGSIZE_MAPA_REGIONAL = (4.5, 5.0)
FIGSIZE_SCATTER = (4.0, 3.5)
FIGSIZE_HEATMAP = (4.5, 4.0)

# rcParams padrão Erneson Oliveira: Computer Modern, ticks para fora, sem grade
ESTILO_ERNESON: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": False,
    "figure.autolayout": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def aplicar_estilo() -> None:
    """Aplica globalmente o estilo visual padrão Erneson Oliveira ao matplotlib."""
    mpl.rcParams.update(ESTILO_ERNESON)


def rotulo_painel(ax: mpl.axes.Axes, label: str, dentro: bool = False, x: float | None = None, y: float | None = None) -> None:
    """
    Adiciona rótulo de painel '(a)', '(b)'... em minúsculas, não negrito, fontsize=12.
    Se dentro=False (gráficos): fora do eixo no canto superior esquerdo.
    Se dentro=True (mapas): dentro do mapa no canto superior esquerdo.
    """
    fmt_label = f"({label.strip('()')})"
    if dentro:
        px = 0.03 if x is None else x
        py = 0.94 if y is None else y
        ax.text(
            px,
            py,
            fmt_label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="normal",
            va="top",
            ha="left",
            zorder=100,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        )
    else:
        px = -0.14 if x is None else x
        py = 1.04 if y is None else y
        ax.text(
            px,
            py,
            fmt_label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="normal",
            va="bottom",
            ha="right",
            zorder=100,
        )


def rotulo_canto(ax: mpl.axes.Axes, texto: str, canto: str = "lower right", fontsize: int = 10, bbox: bool = True) -> None:
    """Insere identificador da região ou base de dados em serifa num canto sem borda."""
    pos_map = {
        "lower right": (0.96, 0.04, "right", "bottom"),
        "lower left": (0.04, 0.04, "left", "bottom"),
        "upper right": (0.96, 0.96, "right", "top"),
        "upper left": (0.04, 0.96, "left", "top"),
    }
    px, py, ha, va = pos_map.get(canto, (0.96, 0.04, "right", "bottom"))
    extra = {}
    if bbox:
        extra["bbox"] = dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
    ax.text(px, py, texto, transform=ax.transAxes, fontsize=fontsize, ha=ha, va=va, zorder=90, **extra)


def anotar_expoente(ax: mpl.axes.Axes, texto: str, canto: str = "lower left", fontsize: int = 9, bbox: bool = True) -> None:
    """
    Exibe texto matemático de parâmetros num canto livre (ex.: gamma = 0.54 +- 0.01, R^2 = 0.87).
    """
    pos_map = {
        "lower left": (0.05, 0.05, "left", "bottom"),
        "lower right": (0.95, 0.05, "right", "bottom"),
        "upper left": (0.05, 0.92, "left", "top"),
        "upper right": (0.95, 0.92, "right", "top"),
    }
    px, py, ha, va = pos_map.get(canto, (0.05, 0.05, "left", "bottom"))
    extra = {}
    if bbox:
        extra["bbox"] = dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.88)
    ax.text(
        px,
        py,
        texto,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=ha,
        va=va,
        zorder=85,
        linespacing=1.3,
        **extra,
    )


def guia_inclinacao(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    dx: float,
    dy: float,
    label: str = r"$\beta = 1$",
    fontsize: int = 9,
    color: str = "black",
) -> None:
    """Desenha colchete indicador de inclinação em 'L' (segmento horizontal e vertical)."""
    # Segmento horizontal de x a x+dx em y
    ax.plot([x, x + dx], [y, y], color=color, lw=1.0, zorder=50)
    # Segmento vertical de y a y+dy em x+dx
    ax.plot([x + dx, x + dx], [y, y + dy], color=color, lw=1.0, zorder=50)
    # Rótulo ao lado
    ax.text(
        x + dx + 0.05 * abs(dx),
        y + dy / 2.0,
        label,
        fontsize=fontsize,
        va="center",
        ha="left",
        color=color,
        zorder=51,
    )


def scatter_vazado(
    ax: mpl.axes.Axes,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    c: str = "C0",
    marker: str = "o",
    s: float = 35.0,
    lw: float = 1.0,
    alpha: float = 0.85,
    label: str | None = None,
    zorder: int = 20,
) -> Any:
    """Plota pontos vazados (facecolors='none') no padrão Erneson Oliveira."""
    return ax.scatter(
        x,
        y,
        edgecolors=c,
        facecolors="none",
        marker=marker,
        s=s,
        linewidths=lw,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def reta_ols(
    ax: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str = "black",
    lw: float = 1.2,
    linestyle: str = "-",
    label: str | None = None,
    zorder: int = 30,
) -> dict[str, float]:
    """
    Ajusta regressão linear ordinária (OLS) e plota linha preta sólida restrita à faixa ajustada.
    Retorna dicionário com coeficientes: slope, intercept, r_value, r2, stderr, p_value.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    xs, ys = x_arr[m], y_arr[m]

    slope, intercept, r_value, p_value, stderr = stats.linregress(xs, ys)
    x_line = np.linspace(xs.min(), xs.max(), 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color=color, lw=lw, linestyle=linestyle, label=label, zorder=zorder)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "r2": float(r_value**2),
        "stderr": float(stderr),
        "p_value": float(p_value),
    }


def curva_nw(
    ax: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float | None = None,
    n_points: int = 80,
    n_boot: int = 200,
    color: str = "C1",
    alpha_band: float = 0.25,
    lw: float = 1.3,
    label: str | None = "Nadaraya–Watson",
    zorder: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ajusta regressão local não-paramétrica de Nadaraya-Watson com núcleo gaussiano
    e calcula intervalo de confiança empírico de 95% via bootstrap.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    xs, ys = x_arr[m], y_arr[m]
    n = len(xs)
    if n < 5:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if bandwidth is None:
        # Regra de Silverman para estimativa de largura de banda
        h = 1.06 * np.std(xs, ddof=1) * (n ** (-1 / 5))
        bandwidth = max(h, 0.15)

    x_eval = np.linspace(xs.min(), xs.max(), n_points)

    def _nw_estimate(x_in, y_in, x_out, bw):
        u = (x_out[:, None] - x_in[None, :]) / bw
        w = np.exp(-0.5 * (u**2))
        w_sum = np.sum(w, axis=1)
        w_sum = np.where(w_sum == 0, 1e-12, w_sum)
        return np.sum(w * y_in[None, :], axis=1) / w_sum

    y_eval = _nw_estimate(xs, ys, x_eval, bandwidth)

    # Bootstrap para banda de confiança 95%
    rng = np.random.default_rng(42)
    boot_preds = np.empty((n_boot, n_points), dtype=float)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_preds[b] = _nw_estimate(xs[idx], ys[idx], x_eval, bandwidth)

    ci_low = np.percentile(boot_preds, 2.5, axis=0)
    ci_high = np.percentile(boot_preds, 97.5, axis=0)

    ax.plot(x_eval, y_eval, color=color, lw=lw, linestyle="--", label=label, zorder=zorder)
    ax.fill_between(x_eval, ci_low, ci_high, color=color, alpha=alpha_band, zorder=zorder - 1)

    return x_eval, y_eval, ci_low, ci_high


def referencia_inclinacao_1(
    ax: mpl.axes.Axes,
    x0: float,
    x1: float,
    y0: float,
    color: str = "#888888",
    lw: float = 1.0,
    linestyle: str = ":",
    label: str | None = r"$\beta = 1$",
) -> None:
    """Plota linha de referência isométrica com inclinação 1 (tracejada/pontilhada cinza)."""
    y1 = y0 + (x1 - x0)
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, linestyle=linestyle, label=label, zorder=15)


def eixos_log10(
    ax: mpl.axes.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Configura os eixos com rótulos matemáticos e moldura completa sem grade."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.tick_params(direction="out", length=4.0, width=0.8, top=False, right=False)


def colorbar_direita(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    im: mpl.cm.ScalarMappable,
    label: str,
    size: str = "3%",
    pad: float = 0.05,
    log: bool = False,
) -> mpl.colorbar.Colorbar:
    """Adiciona colorbar vertical fina à direita do eixo com rótulo girado 90 graus."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(label, rotation=90, labelpad=8)
    cbar.ax.tick_params(direction="out", labelsize=9)
    return cbar


def eixo_mapa(ax: mpl.axes.Axes, extent: tuple[float, float, float, float] | None = None) -> None:
    """Formata eixo para mapas (sem moldura, ticks ou rótulos, aspect equal)."""
    ax.set_aspect("equal")
    ax.set_axis_off()
    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])


def cores_bacias_tab20(basin_array: np.ndarray, ocean_color: str = "#e5e5e5") -> tuple[np.ndarray, ListedColormap]:
    """
    Aplica coloração de grafo com colormap tab20 (20 cores) nas bacias do IPBA,
    garantindo que bacias vizinhas de 4-conectividade nunca compartilhem a mesma cor.
    Retorna a matriz colorida (RGB/RGBA ou índices de cor) e o colormap.
    """
    labels = np.unique(basin_array[basin_array > 0])
    if len(labels) == 0:
        cmap = ListedColormap([ocean_color])
        return np.zeros_like(basin_array, dtype=int), cmap

    # Construir grafo de adjacência de 4-vizinhos
    adj: dict[int, set[int]] = {int(lbl): set() for lbl in labels}
    nrows, ncols = basin_array.shape

    # Adjacência vertical
    diff_v = basin_array[:-1, :] != basin_array[1:, :]
    v_a = basin_array[:-1, :][diff_v]
    v_b = basin_array[1:, :][diff_v]
    for a, b in zip(v_a, v_b):
        if a > 0 and b > 0:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

    # Adjacência horizontal
    diff_h = basin_array[:, :-1] != basin_array[:, 1:]
    h_a = basin_array[:, :-1][diff_h]
    h_b = basin_array[:, 1:][diff_h]
    for a, b in zip(h_a, h_b):
        if a > 0 and b > 0:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

    # Algoritmo Welsh-Powell / Greedy para colorir grafo com até 20 cores (tab20)
    # Ordena nós pelo grau decrescente
    sorted_nodes = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
    color_assign: dict[int, int] = {}
    for node in sorted_nodes:
        neighbor_colors = {color_assign[nbr] for nbr in adj[node] if nbr in color_assign}
        # Escolhe a menor cor de 1 a 20 não usada pelos vizinhos
        c = 1
        while c in neighbor_colors:
            c += 1
            if c > 20:
                c = ((c - 1) % 20) + 1  # Fallback de emergência caso haja clique > 20
        color_assign[node] = c

    # Validar se vizinhos têm cores diferentes
    conflicts = 0
    for node, neighbors in adj.items():
        for nbr in neighbors:
            if color_assign[node] == color_assign[nbr]:
                conflicts += 1
    # Asserção de integridade conforme Diretriz 01
    assert conflicts == 0, f"Conflito de coloração de bacias vizinhas: {conflicts} conflitos detectados!"

    # Mapear para matriz
    colored_grid = np.zeros_like(basin_array, dtype=int)
    for lbl in labels:
        colored_grid[basin_array == lbl] = color_assign[int(lbl)]

    tab20_base = mpl.colormaps["tab20"].colors
    # Índice 0: cor do oceano (#e5e5e5)
    full_colors = [ocean_color] + list(tab20_base)
    cmap = ListedColormap(full_colors)

    return colored_grid, cmap


def cores_manchas(n_clusters: int = 50) -> ListedColormap:
    """Gera paleta discreta distinta para manchas urbanas do CCA com fundo branco."""
    tab20 = mpl.colormaps["tab20"].colors
    rep = int(np.ceil(n_clusters / 20)) + 1
    palette = ["#ffffff"] + list(tab20) * rep
    return ListedColormap(palette[: n_clusters + 1])


def inset_localizador(
    ax_main: mpl.axes.Axes,
    bbox: tuple[float, float, float, float],
    canto: str = "lower left",
    width: float = 0.28,
    height: float = 0.24,
    geojson_path: Path = WORLD_GEOJSON,
) -> mpl.axes.Axes:
    """
    Insere mapa-múndi localizador com a bounding box realçada em retângulo vermelho/preto tracejado.
    bbox: (lon_min, lat_min, lon_max, lat_max)
    """
    pos_map = {
        "lower left": (0.02, 0.02),
        "lower right": (1.0 - width - 0.02, 0.02),
        "upper left": (0.02, 1.0 - height - 0.02),
        "upper right": (1.0 - width - 0.02, 1.0 - height - 0.02),
    }
    x0, y0 = pos_map.get(canto, (0.02, 0.02))

    ax_inset = ax_main.inset_axes([x0, y0, width, height])
    ax_inset.set_aspect("equal")
    ax_inset.set_axis_off()
    ax_inset.set_xlim(-180, 180)
    ax_inset.set_ylim(-60, 85)

    # Fundo oceano do inset branco
    bg = Rectangle((-180, -60), 360, 145, facecolor="#ffffff", edgecolor="#aaaaaa", lw=0.6, zorder=1)
    ax_inset.add_patch(bg)

    # Desenhar continentes do GeoJSON se disponível
    if geojson_path.is_file():
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
            for feat in gj.get("features", []):
                geom = feat.get("geometry", {})
                gtype = geom.get("type")
                coords = geom.get("coordinates", [])
                if gtype == "Polygon":
                    for ring in coords:
                        poly = Polygon(ring, facecolor="#d0d0d0", edgecolor="none", zorder=2)
                        ax_inset.add_patch(poly)
                elif gtype == "MultiPolygon":
                    for poly_coords in coords:
                        for ring in poly_coords:
                            poly = Polygon(ring, facecolor="#d0d0d0", edgecolor="none", zorder=2)
                            ax_inset.add_patch(poly)
        except Exception:
            pass

    # Desenhar retângulo do recorte
    lon_min, lat_min, lon_max, lat_max = bbox
    rect = Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        facecolor="none",
        edgecolor="#d62728",
        lw=1.0,
        linestyle="-",
        zorder=10,
    )
    ax_inset.add_patch(rect)

    return ax_inset


def salvar(fig: mpl.figure.Figure, basepath: str | Path, dpi: int = 300) -> tuple[Path, Path]:
    """
    Exporta a figura simultaneamente em PNG (300 DPI) e PDF vetorial ao lado do caminho base.
    """
    base = Path(basepath)
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.parent / base.stem
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return png_path, pdf_path
