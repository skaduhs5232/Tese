"""
pipeline/notebook_utils.py
--------------------------
Funcoes curtas usadas pelos notebooks:
  - carrega os tipos de lixo da NOAA (com cota GEBCO) e as bacias (GebcoCutter + IPBA + CCA);
  - media dos pontos num circulo de raio R em volta de cada sink;
  - lei de potencia log-log (nuvem + medias por faixa) e mosaicos de graficos.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy import stats
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from geracao_graficos.estilo_erneson import aplicar_estilo, rotulo_painel
from pipeline.gebco_cutter import (
    REPO_ROOT,
    executar_percolacao_gebco_cutter,
    processar_porcao_terra_gebco_cutter,
    recortar_cluster_gebco_cutter,
)
from pipeline.ipba_runner import EARTH_RADIUS_KM

NOAA_GEOJSON = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
GEBCO_TIF = REPO_ROOT / "dados" / "gebco" / "gebco_2026_sub_ice_geotiff.tif"
GPW_TIF = REPO_ROOT / "dados" / "gpw_data" / "gpw_v4_population_count_rev11_2020_30_sec.tif"
CACHE_PERC = REPO_ROOT / "relatorios" / "cache_percolacao_gebco_h0.npz"
PORCOES_DIR = REPO_ROOT / "relatorios" / "porcoes_terra"
FIGURAS_DIR = REPO_ROOT / "relatorios" / "figuras"

UNIDADES = {
    "Água oceânica": "pieces/m³",
    "Areia de praia": "pieces/m³",
    "Sedimento oceânico": "pieces/kg",
    "Nurdle Patrol": "pieces/10 min",
}
EIXOS_X = {"total_population": "População da Bacia", "area_km2": "Área da Bacia (km²)"}
CORES = ["#0284c7", "#d97706", "#7c3aed", "#059669"]


def to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Converte coordenadas geograficas (graus) em vetores unitarios 3D."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def carregar_noaa_por_tipo() -> dict[str, pd.DataFrame]:
    """Le o GeoJSON da NOAA, anota a cota GEBCO e separa os tipos de lixo (nomes em portugues)."""
    with open(NOAA_GEOJSON, "r", encoding="utf-8") as f:
        gj = json.load(f)

    df = pd.DataFrame([ft["properties"] for ft in gj["features"]])
    df["lat"] = pd.to_numeric(df["Latitude__degree_"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude_degree_"], errors="coerce")
    df["mesh_mm"] = pd.to_numeric(df["Mesh_size__mm_"], errors="coerce")
    df["mp_val"] = pd.to_numeric(df["Microplastics_measurement"], errors="coerce")
    df["nurdle_val"] = pd.to_numeric(df["Standardized_Nurdle__Amount"], errors="coerce")
    df["valor"] = np.where(df["Medium"] == "Nurdle Patrol", df["nurdle_val"], df["mp_val"])
    df = df[df["lat"].notna() & df["lon"].notna() & df["valor"].notna()].copy().reset_index(drop=True)

    gdal.UseExceptions()
    ds = gdal.Open(str(GEBCO_TIF), gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    dem = ds.GetRasterBand(1).ReadAsArray()
    cols = np.clip(np.floor((df["lon"].to_numpy() - gt[0]) / gt[1]).astype(int), 0, ds.RasterXSize - 1)
    rows = np.clip(np.floor((gt[3] - df["lat"].to_numpy()) / abs(gt[5])).astype(int), 0, ds.RasterYSize - 1)
    df["gebco_h"] = dem[rows, cols].astype(float)
    del ds

    mar = df["gebco_h"] <= 0.0
    return {
        "Água oceânica": df[(df["Medium"] == "Ocean water") & (df["Unit"] == "pieces/m3") & df["mesh_mm"].between(0.02, 5.0) & mar].reset_index(drop=True),
        "Areia de praia": df[(df["Medium"] == "Beach") & (df["Unit"] == "pieces/m3")].reset_index(drop=True),
        "Sedimento oceânico": df[(df["Medium"] == "Ocean sediment") & mar].reset_index(drop=True),
        "Nurdle Patrol": df[df["Medium"] == "Nurdle Patrol"].reset_index(drop=True),
    }


def identificar_porcoes_ativas(
    pontos_df: pd.DataFrame,
    max_r_km: float = 50.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[int]]:
    """Segmenta o globo via `externos/GebcoCutter` e lista as porcoes de terra proximas aos pontos."""
    info, arr, label, vals, _ = executar_percolacao_gebco_cutter(GEBCO_TIF, h_star=0.0, cache_npz=CACHE_PERC)
    gt = info["GeoTransform"]
    p = arr > 0
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    coast = p & binary_dilation(~p, structure=struct4)
    sr, sc = np.nonzero(coast)
    slats = gt[3] + (sr + 0.5) * gt[5]
    slons = ((gt[0] + (sc + 0.5) * gt[1] + 180.0) % 360.0) - 180.0

    tree_p = cKDTree(to_xyz(pontos_df["lat"].to_numpy(), pontos_df["lon"].to_numpy()))
    chord = float(2.0 * np.sin((max_r_km / EARTH_RADIUS_KM) / 2.0))
    has_pts = np.array([len(x) > 0 for x in tree_p.query_ball_point(to_xyz(slats, slons), r=chord)])
    ativos_set = {int(v) for v in np.unique(label[sr, sc][has_pts])}
    porcoes_ordenadas = [int(v) for v in vals if int(v) in ativos_set]
    return info, arr, label, porcoes_ordenadas


def processar_ou_carregar_porcao(
    val: int,
    info: dict[str, Any],
    arr: np.ndarray,
    label: np.ndarray,
) -> pd.DataFrame:
    """Executa (ou le do cache) `externos/IPBA` + `externos/CCA` para uma porcao de terra `val`."""
    PORCOES_DIR.mkdir(parents=True, exist_ok=True)
    cache_csv = PORCOES_DIR / f"bacias_porcao_{int(val)}.csv"
    if cache_csv.is_file() and cache_csv.stat().st_size > 10:
        return pd.read_csv(cache_csv)
    arr_c, cl_c, info_c, jmin, jmax, cols = recortar_cluster_gebco_cutter(info, arr, label, int(val))
    return processar_porcao_terra_gebco_cutter(int(val), arr_c, cl_c, info_c, jmin, jmax, cols, GPW_TIF, out_dir=PORCOES_DIR)


def carregar_bacias(pontos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """100% das bacias (IPBA + CCA) das porcoes de terra a ate 50 km de algum ponto."""
    info, arr, label, porcoes = identificar_porcoes_ativas(pd.concat(pontos.values(), ignore_index=True))
    dfs = [processar_ou_carregar_porcao(v, info, arr, label) for v in porcoes]
    return pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)


def recorte_top95_global(bacias: pd.DataFrame) -> pd.DataFrame:
    """Maiores bacias que somam 95% da area total (recorte classico do IPBA)."""
    b = bacias.sort_values("area_km2", ascending=False).reset_index(drop=True)
    acum = b["area_km2"].cumsum() / b["area_km2"].sum()
    return b.iloc[: int((acum <= 0.95).sum()) + 1].reset_index(drop=True)


def filtrar_pop(df: pd.DataFrame, pop_min: float = 0) -> pd.DataFrame:
    """Filtra bacias com populacao total positiva e >= `pop_min`."""
    return df[(df["total_population"] > 0) & (df["total_population"] >= pop_min)]


def media_por_sink_raio(bacias: pd.DataFrame, pontos: pd.DataFrame, r_km: float) -> pd.DataFrame:
    """Media dos pontos a ate `r_km` de cada sink; devolve so as bacias com media > 0."""
    tree = cKDTree(to_xyz(pontos["lat"].to_numpy(), pontos["lon"].to_numpy()))
    corda = 2.0 * np.sin(r_km / EARTH_RADIUS_KM / 2.0)
    vizinhos = tree.query_ball_point(to_xyz(bacias["latitude"].to_numpy(), bacias["longitude"].to_numpy()), r=corda)
    vals = pontos["valor"].to_numpy()
    out = bacias.copy()
    out["mp_mean"] = [vals[v].mean() if v else np.nan for v in vizinhos]
    out["n_points"] = [len(v) for v in vizinhos]
    return out[out["mp_mean"] > 0].reset_index(drop=True)


def pareados_por_raio(
    bacias: pd.DataFrame,
    pontos: dict[str, pd.DataFrame],
    raios: Any,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Associa a media por raio para cada conjunto de pontos."""
    return {r: {nome: media_por_sink_raio(bacias, pts, r) for nome, pts in pontos.items()} for r in raios}


def _medias_bins(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Media de log x e log y em 8 faixas iguais de log x (so faixas com >= 3 pontos)."""
    faixa = pd.cut(x, np.linspace(x.min(), x.max(), 9), include_lowest=True)
    b = pd.DataFrame({"x": x, "y": y, "f": faixa}).groupby("f", observed=True).agg(n=("y", "size"), xm=("x", "mean"), ym=("y", "mean"))
    return b[b["n"] >= 3]


def ajustar_lei_potencia(df: pd.DataFrame, x_col: str, y_col: str = "mp_mean") -> dict[str, Any]:
    """OLS em log10(x) vs log10(y) na nuvem + R² e beta (ponderado por n) nas medias por faixa."""
    sub = df[(df[x_col] > 0) & (df[y_col] > 0)]
    res: dict[str, Any] = dict(
        N=len(sub), beta=np.nan, intercept=np.nan, se=np.nan,
        r2_pct=np.nan, p_value=np.nan, rho=np.nan, r2_bins_pct=np.nan, beta_bins_w=np.nan,
    )
    if len(sub) < 4:
        return res
    x, y = np.log10(sub[x_col].to_numpy()), np.log10(sub[y_col].to_numpy())
    fit = stats.linregress(x, y)
    res.update(
        beta=fit.slope, intercept=fit.intercept, se=fit.stderr,
        r2_pct=100 * fit.rvalue**2, p_value=fit.pvalue, rho=stats.spearmanr(x, y)[0],
    )
    b = _medias_bins(x, y) if len(sub) >= 20 else pd.DataFrame()
    if len(b) >= 4:
        res["r2_bins_pct"] = 100 * stats.linregress(b["xm"], b["ym"]).rvalue ** 2
        res["beta_bins_w"] = np.polyfit(b["xm"], b["ym"], 1, w=np.sqrt(b["n"]))[0]
    return res


def tabela_ajustes(pareados_por_r: dict[int, dict[str, pd.DataFrame]], pop_mins: Any = (0,)) -> pd.DataFrame:
    """Consolida os ajustes de lei de potencia por conjunto, raio, corte de populacao e eixo X."""
    linhas = []
    for r, pareados in pareados_por_r.items():
        for nome, df in pareados.items():
            for pop_min in pop_mins:
                for x_col in EIXOS_X:
                    res = ajustar_lei_potencia(filtrar_pop(df, pop_min), x_col)
                    res.pop("intercept")
                    linhas.append({"Conjunto": nome, "R_km": r, "Pop_min": pop_min, "X": x_col, **res})
    return pd.DataFrame(linhas)


def plotar_lei_potencia(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    titulo: str,
    xlabel: str,
    ylabel: str,
    cor: str = "#0284c7",
) -> None:
    """Nuvem log-log, reta OLS, medias por faixa (pontos pretos) e caixa com as estatisticas."""
    res = ajustar_lei_potencia(df, x_col)
    ax.set(title=titulo, xlabel=xlabel, ylabel=ylabel)
    if res["N"] < 4:
        ax.text(0.5, 0.5, f"N = {res['N']} (insuficiente)", ha="center", va="center", transform=ax.transAxes)
        return
    sub = df[(df[x_col] > 0) & (df["mp_mean"] > 0)]
    x, y = np.log10(sub[x_col].to_numpy()), np.log10(sub["mp_mean"].to_numpy())
    ax.scatter(x, y, s=22, facecolors="none", edgecolors=cor, alpha=0.7, label=f"Bacias (N = {res['N']})")
    xl = np.array([x.min(), x.max()])
    ax.plot(xl, res["intercept"] + res["beta"] * xl, color=cor, lw=2, label="Ajuste na nuvem")
    if res["N"] >= 20:
        b = _medias_bins(x, y)
        ax.plot(b["xm"], b["ym"], "ko", ms=5, label="Média por faixa")
    bins_txt = (
        f"\nR²_bins = {res['r2_bins_pct']:.1f}% | β_bins (pond) = {res['beta_bins_w']:+.2f}"
        if np.isfinite(res["r2_bins_pct"])
        else ""
    )
    texto = (
        f"β = {res['beta']:+.2f} ± {res['se']:.2f}\n"
        f"R² = {res['r2_pct']:.1f}% (p = {res['p_value']:.1e})\n"
        f"ρ Spearman = {res['rho']:+.2f}{bins_txt}"
    )
    ax.text(
        0.03, 0.97, texto.replace(".", ","), transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cbd5e1"),
    )
    span = max(y.max() - y.min(), 2.0)
    ax.set_ylim(y.min() - 0.15 * span, y.max() + 0.55 * span)
    ax.legend(loc="lower right", fontsize=7)


def plotar_mosaico(
    pareados: dict[str, pd.DataFrame],
    titulo: str,
    pop_min: float = 0,
    arquivo: str | None = None,
) -> None:
    """Linhas = conjuntos (ex.: Agua oceanica, Areia de praia); colunas = Populacao e Area da bacia."""
    aplicar_estilo()
    fig, axes = plt.subplots(len(pareados), 2, figsize=(12, 4.3 * len(pareados)), squeeze=False)
    for i, (nome, df) in enumerate(pareados.items()):
        for j, (x_col, nome_x) in enumerate(EIXOS_X.items()):
            ax = axes[i, j]
            plotar_lei_potencia(
                ax, filtrar_pop(df, pop_min), x_col, nome,
                f"log10({nome_x})", f"log10(Média {UNIDADES[nome]})", CORES[i],
            )
            rotulo_painel(ax, f"({'abcdefgh'[2 * i + j]})", x=-0.12, y=1.04)
    filtro = f"População ≥ {pop_min:,} hab.".replace(",", ".") if pop_min else "sem filtro de população"
    fig.suptitle(f"{titulo} — {filtro}")
    fig.tight_layout()
    if arquivo:
        FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURAS_DIR / f"{arquivo}.png", dpi=200, bbox_inches="tight")
    plt.show()


def plotar_varredura(tab: pd.DataFrame, min_n: int = 20) -> None:
    """Para cada conjunto: R² (%) vs R (Populacao e Area) e N de bacias vs R."""
    aplicar_estilo()
    nomes = tab["Conjunto"].unique()
    fig, axes = plt.subplots(len(nomes), 2, figsize=(12, 3.2 * len(nomes)), sharex=True, squeeze=False)
    for i, nome in enumerate(nomes):
        for x_col, nome_x in EIXOS_X.items():
            s = tab[(tab["Conjunto"] == nome) & (tab["X"] == x_col)]
            axes[i, 0].plot(s["R_km"], s["r2_pct"].where(s["N"] >= min_n), "o-", label=nome_x)
        axes[i, 1].bar(s["R_km"], s["N"], color=CORES[i])
        axes[i, 0].set(title=f"{nome} — R² na nuvem (N ≥ {min_n})", ylabel="R² (%)")
        axes[i, 1].set(title=f"{nome} — bacias com valor no sink", ylabel="N")
        axes[i, 0].legend(fontsize=8)
        for ax in axes[i]:
            for r in (5, 22):
                ax.axvline(r, color="gray", ls=":", lw=1)
    for ax in axes[-1]:
        ax.set_xlabel("Raio R no sink (km)")
        ax.set_xticks(range(1, 23))
    fig.tight_layout()
    plt.show()
