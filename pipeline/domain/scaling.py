"""Filtros, pareamento por raio fixo no sink e ajuste de lei de potência log-log."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from ..config import EIXOS_X, REDES_SUPERFICIE
from .geometry import km_para_corda, to_xyz


def filtrar_redes_superficie(df: pd.DataFrame, metodos: tuple[str, ...] = REDES_SUPERFICIE) -> pd.DataFrame:
    """Mantém apenas coletas feitas com redes de superfície (ex.: Neuston, Manta, AVANI)."""
    return df[df["Sampling_Method"].isin(metodos)].reset_index(drop=True)


def filtrar_pop(df: pd.DataFrame, pop_min: float = 0) -> pd.DataFrame:
    """Filtra bacias com população total positiva e >= pop_min."""
    return df[(df["total_population"] > 0) & (df["total_population"] >= pop_min)]


def media_por_sink_raio(bacias: pd.DataFrame, pontos: pd.DataFrame, r_km: float) -> pd.DataFrame:
    """Média aritmética dos pontos a até r_km de cada sink; devolve só bacias com média > 0."""
    tree = cKDTree(to_xyz(pontos["lat"].to_numpy(), pontos["lon"].to_numpy()))
    vizinhos = tree.query_ball_point(
        to_xyz(bacias["latitude"].to_numpy(), bacias["longitude"].to_numpy()),
        r=km_para_corda(r_km),
    )
    vals = pontos["valor"].to_numpy()
    out = bacias.copy()
    out["mp_mean"] = [vals[v].mean() if v else np.nan for v in vizinhos]
    out["n_points"] = [len(v) for v in vizinhos]
    return out[out["mp_mean"] > 0].reset_index(drop=True)


def pareados_por_raio(
    bacias: pd.DataFrame,
    pontos: dict[str, pd.DataFrame],
    raios: Iterable[int],
) -> dict[int, dict[str, pd.DataFrame]]:
    """Associa a média por raio para cada conjunto de pontos."""
    return {r: {nome: media_por_sink_raio(bacias, pts, r) for nome, pts in pontos.items()} for r in raios}


def medias_bins(x: np.ndarray, y: np.ndarray, n_bins: int = 8, min_por_bin: int = 3) -> pd.DataFrame:
    """Média de log(x) e log(y) em n_bins faixas iguais de log(x) com >= min_por_bin pontos."""
    faixas = pd.cut(x, np.linspace(x.min(), x.max(), n_bins + 1), include_lowest=True)
    agg = pd.DataFrame({"x": x, "y": y, "f": faixas}).groupby("f", observed=True).agg(
        n=("y", "size"), xm=("x", "mean"), ym=("y", "mean")
    )
    return agg[agg["n"] >= min_por_bin]


def ajustar_lei_potencia(df: pd.DataFrame, x_col: str, y_col: str = "mp_mean") -> dict[str, Any]:
    """OLS em log10(x) vs log10(y) na nuvem + Spearman + R² e beta ponderado nos bins."""
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
    b = medias_bins(x, y) if len(sub) >= 20 else pd.DataFrame()
    if len(b) >= 4:
        res["r2_bins_pct"] = 100 * stats.linregress(b["xm"], b["ym"]).rvalue ** 2
        res["beta_bins_w"] = np.polyfit(b["xm"], b["ym"], 1, w=np.sqrt(b["n"]))[0]
    return res


def tabela_ajustes(
    pareados_por_r: dict[int, dict[str, pd.DataFrame]],
    pop_mins: Iterable[float] = (0,),
) -> pd.DataFrame:
    """Consolida os ajustes de lei de potência por conjunto, raio, corte de população e eixo X."""
    linhas = []
    for r, pareados in pareados_por_r.items():
        for nome, df in pareados.items():
            for pop_min in pop_mins:
                for x_col in EIXOS_X:
                    res = ajustar_lei_potencia(filtrar_pop(df, pop_min), x_col)
                    res.pop("intercept")
                    linhas.append({"Conjunto": nome, "R_km": r, "Pop_min": pop_min, "X": x_col, **res})
    return pd.DataFrame(linhas)
