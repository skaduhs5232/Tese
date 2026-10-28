"""
estatistica.py
--------------
Implementação manual em NumPy das métricas estatísticas, pareamento geodésico foz-coleta,
regressão alométrica (OLS em log10), bootstrap de pares, teste de permutação e leave-one-out.
Zero dependências de `scipy.stats` ou bibliotecas estatísticas externas.
"""

import numpy as np
from numba import njit
from ..geometria_esferica.area_esferica import (
    R_EARTH_KM,
    haversine_deg,
    haversine_rad,
    pixel_area_by_latitude,
)


def _inclinacao(X: np.ndarray, Y: np.ndarray):
    Xc = X - X.mean(axis=-1, keepdims=True)
    Yc = Y - Y.mean(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc * Yc).sum(axis=-1) / (Xc * Xc).sum(axis=-1)


def _postos_com_empates(x: np.ndarray) -> np.ndarray:
    """
    Calcula os postos (ranks) médios de um vetor 1D tratando empates
    exatamente como o ranking estatístico padrão, usando apenas NumPy.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    ordem = np.argsort(x, kind="mergesort")
    xs = x[ordem]
    dif = np.concatenate(([True], xs[1:] != xs[:-1], [True]))
    idx = np.flatnonzero(dif)
    contagens = np.diff(idx)
    postos_ordenados = np.repeat((idx[:-1] + idx[1:] + 1.0) * 0.5, contagens)
    postos = np.empty(n, dtype=np.float64)
    postos[ordem] = postos_ordenados
    return postos


def _erfc_aprox(x: np.ndarray) -> np.ndarray:
    """Aproximação de Abramowitz & Stegun (7.1.26) para erfc(|x|) em NumPy puro (erro < 1.5e-7)."""
    z = np.abs(np.asarray(x, dtype=np.float64))
    t = 1.0 / (1.0 + 0.3275911 * z)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    res = poly * np.exp(-z * z)
    return np.where(x >= 0, res, 2.0 - res)


def _p_valor_t_student(t_abs: float, df: int) -> float:
    """
    Aproximação de Peizer-Pratt para o p-valor bicaudal da distribuição t de Student
    com `df` graus de liberdade em NumPy puro.
    """
    if df <= 0 or not np.isfinite(t_abs):
        return np.nan
    if t_abs == 0.0:
        return 1.0
    if df == 1:
        return float(2.0 * (0.5 - np.arctan(t_abs) / np.pi))
    if df == 2:
        return float(1.0 - t_abs / np.sqrt(2.0 + t_abs * t_abs))
    # Transformação normal de Peizer-Pratt / Wallace para t de Student
    z = np.sqrt(df * np.log(1.0 + (t_abs * t_abs) / df) * (1.0 - 1.0 / (2.0 * df)))
    return float(np.clip(_erfc_aprox(z / np.sqrt(2.0)), 0.0, 1.0))


def pearson_r(x: np.ndarray, y: np.ndarray):
    """
    Calcula o coeficiente de correlação linear de Pearson (r) e seu p-valor bicaudal
    usando exclusivamente NumPy.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 3:
        return np.nan, np.nan
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
    if denom <= 0:
        return 0.0, 1.0
    r = float(np.clip(np.sum(xc * yc) / denom, -1.0, 1.0))
    if abs(r) >= 1.0 - 1e-15:
        return r, 0.0
    df = n - 2
    t_stat = abs(r) * np.sqrt(df / max(1e-15, 1.0 - r * r))
    return r, _p_valor_t_student(t_stat, df)


def spearman_rho(x: np.ndarray, y: np.ndarray):
    """
    Calcula o coeficiente de correlação de postos de Spearman (rho_s) e seu p-valor
    usando exclusivamente NumPy.
    """
    rx = _postos_com_empates(x)
    ry = _postos_com_empates(y)
    return pearson_r(rx, ry)


def ajuste_escala_log_log(x: np.ndarray, y: np.ndarray, n_boot: int = 10_000, semente: int = 0):
    """
    Ajusta a lei de escala alométrica y = c * x^beta por Mínimos Quadrados Ordinários (OLS)
    em log10, com intervalo de confiança de 95% via bootstrap de pares e p-valor de permutação
    para a hipótese nula beta = 0, escrito 100% em NumPy.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx = np.log10(x[m])
    ly = np.log10(y[m])
    n = lx.size
    if n < 3:
        return dict(n=int(n), beta=np.nan, log10_c=np.nan, ic95_inf=np.nan, ic95_sup=np.nan,
                    p_perm=np.nan, r=np.nan, p_r=np.nan, rho=np.nan, p_rho=np.nan, r2=np.nan)

    beta = float(_inclinacao(lx, ly))
    a = float(ly.mean() - beta * lx.mean())

    rng = np.random.default_rng(semente)
    idx = rng.integers(0, n, size=(n_boot, n))
    bb = _inclinacao(lx[idx], ly[idx])
    aa = ly[idx].mean(axis=-1) - bb * lx[idx].mean(axis=-1)
    ok = np.isfinite(bb)

    perm = _inclinacao(
        np.broadcast_to(lx, (n_boot, n)),
        rng.permuted(np.tile(ly, (n_boot, 1)), axis=1),
    )
    xs = np.linspace(lx.min(), lx.max(), 100)
    if ok.any():
        banda = np.percentile(aa[ok, None] + bb[ok, None] * xs[None, :], [2.5, 97.5], axis=0)
        ic95_inf = float(np.percentile(bb[ok], 2.5))
        ic95_sup = float(np.percentile(bb[ok], 97.5))
    else:
        banda = np.full((2, 100), np.nan)
        ic95_inf, ic95_sup = np.nan, np.nan

    p_perm = float((np.sum(np.abs(perm) >= abs(beta)) + 1) / (n_boot + 1))
    r, p_r = pearson_r(lx, ly)
    rho, p_rho = spearman_rho(lx, ly)

    return dict(
        n=int(n),
        beta=beta,
        log10_c=a,
        ic95_inf=ic95_inf,
        ic95_sup=ic95_sup,
        p_perm=p_perm,
        r=float(r),
        p_r=float(p_r),
        rho=float(rho),
        p_rho=float(p_rho),
        r2=float(r * r),
        xs=xs,
        ys=a + beta * xs,
        banda=banda,
    )


def leave_one_out_alometria(x: np.ndarray, y: np.ndarray, basin_ids: np.ndarray = None):
    """
    Análise de sensibilidade leave-one-out em log10 usando apenas NumPy:
    recalcula r de Pearson e o expoente beta removendo uma bacia por vez.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lx = np.log10(x)
    ly = np.log10(y)
    n = lx.size
    if basin_ids is None:
        basin_ids = np.arange(1, n + 1)

    r_all, _ = pearson_r(lx, ly)
    beta_all = float(_inclinacao(lx, ly))

    r_sem = np.empty(n, dtype=np.float64)
    beta_sem = np.empty(n, dtype=np.float64)
    for i in range(n):
        lx_i = np.delete(lx, i)
        ly_i = np.delete(ly, i)
        r_sem[i], _ = pearson_r(lx_i, ly_i)
        beta_sem[i] = float(_inclinacao(lx_i, ly_i))

    return dict(
        basin_id=np.asarray(basin_ids),
        pop_total=x,
        r_all=r_all,
        beta_all=beta_all,
        r_sem=r_sem,
        beta_sem=beta_sem,
        delta_r=r_sem - r_all,
        delta_beta=beta_sem - beta_all,
    )


def parear_fozes_e_sites(
    sink_ids: np.ndarray,
    sink_lons: np.ndarray,
    sink_lats: np.ndarray,
    dlon_deg: float,
    dlat_deg: float,
    site_lons: np.ndarray,
    site_lats: np.ndarray,
    R_k: float = R_EARTH_KM,
    bloco_sinks: int = 1000,
):
    """
    Para cada foz de bacia principal (`sink_ids`, centrada em `sink_lons`, `sink_lats`
    com largura de pixel `dlon_deg x dlat_deg`), encontra o site de coleta mais próximo
    usando a fórmula de Haversine (Eq. 4 de Oliveira et al., 2019) em blocos de memória controlada.

    Um site dentro do próprio pixel da foz tem distância 0 km.

    Retorna:
      nearest_site_idx : índice do site mais próximo para cada sink (shape: n_sinks,)
      nearest_dist_km  : distância em km de cada sink ao seu site mais próximo (shape: n_sinks,)
      site_min_dist_km : menor distância de cada site até alguma foz principal (shape: n_sites,)
    """
    sink_lons = np.asarray(sink_lons, dtype=np.float64)
    sink_lats = np.asarray(sink_lats, dtype=np.float64)
    site_lons = np.asarray(site_lons, dtype=np.float64)
    site_lats = np.asarray(site_lats, dtype=np.float64)

    n_sinks = sink_lons.size
    n_sites = site_lons.size
    if n_sinks == 0 or n_sites == 0:
        return (
            np.zeros(n_sinks, dtype=np.int64),
            np.full(n_sinks, np.inf),
            np.full(n_sites, np.inf),
        )

    half_dx = 0.5 * abs(float(dlon_deg))
    half_dy = 0.5 * abs(float(dlat_deg))

    nearest_site_idx = np.empty(n_sinks, dtype=np.int64)
    nearest_dist_km = np.empty(n_sinks, dtype=np.float64)
    site_min_dist_km = np.full(n_sites, np.inf, dtype=np.float64)

    s_lon_row = site_lons[None, :]
    s_lat_row = site_lats[None, :]

    for i0 in range(0, n_sinks, bloco_sinks):
        i1 = min(n_sinks, i0 + bloco_sinks)
        k_lon = sink_lons[i0:i1, None]
        k_lat = sink_lats[i0:i1, None]

        # Ponto do pixel-sink mais próximo do site
        clip_lon = np.clip(s_lon_row, k_lon - half_dx, k_lon + half_dx)
        clip_lat = np.clip(s_lat_row, k_lat - half_dy, k_lat + half_dy)

        D_bloco = haversine_deg(clip_lon, clip_lat, s_lon_row, s_lat_row, R_k=R_k)
        j_min = np.argmin(D_bloco, axis=1)
        nearest_site_idx[i0:i1] = j_min
        nearest_dist_km[i0:i1] = D_bloco[np.arange(i1 - i0), j_min]
        site_min_dist_km = np.minimum(site_min_dist_km, D_bloco.min(axis=0))

    return nearest_site_idx, nearest_dist_km, site_min_dist_km


# ==============================================================================
# DENSIDADE POPULACIONAL ESFÉRICA, CALIBRAÇÃO DE ZIPF E SOMA DE PESSOAS POR BACIA
# ==============================================================================

def calcular_densidade_gpw(
    pop_count: np.ndarray,
    lat_centers_deg: np.ndarray,
    dlon_deg: float,
    dlat_deg: float,
    R_k: float = R_EARTH_KM,
):
    """
    Converte a matriz `pop_count` (contagem de habitantes por pixel do GPW v4)
    em densidade populacional (hab/km^2), dividindo pela área de dois triângulos
    esféricos de cada linha de latitude (Eq. 3 e Eq. 4 de Oliveira et al., 2019).
    """
    pop = np.nan_to_num(np.asarray(pop_count, dtype=np.float64), nan=0.0)
    pop = np.where(pop > 0.0, pop, 0.0)
    row_areas_km2 = pixel_area_by_latitude(lat_centers_deg, dlon_deg, dlat_deg, R_k=R_k)
    dens_km2 = pop / row_areas_km2[:, None]
    return dens_km2, row_areas_km2


def vizinhanca_haversine(
    lat_centers_deg: np.ndarray,
    dlon_deg: float,
    dlat_deg: float,
    l_km: float,
    R_k: float = R_EARTH_KM,
):
    """
    Pré-computa os deslocamentos (di, dj) de vizinhos na grade cujos centros distam
    menos de `l_km` segundo a fórmula de Haversine (Eq. 4 de Oliveira et al., 2019).
    """
    lat = np.asarray(lat_centers_deg, dtype=np.float64)
    dx = abs(float(dlon_deg))
    dy = abs(float(dlat_deg))
    dlat_rad = np.radians(dy)
    dlon_rad = np.radians(dx)

    max_lat_deg = min(89.9, float(np.abs(lat).max()) + dy)
    ri = int(np.ceil(l_km / (R_k * dlat_rad)))
    rj = int(np.ceil(l_km / (R_k * dlon_rad * max(0.05, np.cos(np.radians(max_lat_deg))))))

    di_grid, dj_grid = np.meshgrid(np.arange(-ri, ri + 1), np.arange(-rj, rj + 1), indexing="ij")
    m = (di_grid != 0) | (dj_grid != 0)
    di = di_grid[m]
    dj = dj_grid[m]

    step_lat_rad = np.radians(lat[1] - lat[0]) if lat.size > 1 else -dlat_rad
    phi_0 = np.radians(lat)[:, None]
    phi_1 = phi_0 + di[None, :] * step_lat_rad
    lam_0 = np.zeros_like(phi_0)
    lam_1 = dj[None, :] * dlon_rad

    dist = haversine_rad(phi_0, lam_0, phi_1, lam_1, R_k=R_k)
    valido = dist < l_km
    u = valido.any(axis=0)
    return di[u].astype(np.int64), dj[u].astype(np.int64), np.ascontiguousarray(valido[:, u])


def escala_minima_l_metros(
    lat_centers_deg: np.ndarray,
    dlon_deg: float,
    dlat_deg: float,
    R_k: float = R_EARTH_KM,
) -> int:
    """
    Calcula o limiar mínimo l_min (em metros, arredondado para centena superior)
    acima da maior diagonal entre pixels vizinhos na faixa de latitude do recorte.
    """
    lat = np.asarray(lat_centers_deg, dtype=np.float64)
    i_eq = int(np.abs(lat).argmin())
    phi_0 = np.radians(lat[i_eq])
    phi_1 = np.radians(lat[i_eq] + abs(float(dlat_deg)))
    diag_km = float(haversine_rad(phi_0, 0.0, phi_1, np.radians(abs(float(dlon_deg))), R_k=R_k))
    return int(np.ceil(diag_km * 10.0) * 100)


def alpha_hill(x: np.ndarray, n: int) -> float:
    """
    Estimador de Hill para o expoente de Zipf alpha_N nas N maiores manchas:
      alpha_N = (1 / (N - 1)) * sum_{i=1}^N ln(S_i / S_N)
    """
    if x.size < n or n < 2:
        return np.nan
    s = np.partition(x, x.size - n)[-n:]
    s_min = s.min()
    if s_min <= 0:
        return np.nan
    return float(np.sum(np.log(s / s_min)) / (n - 1))


def ajuste_csn_zipf(pop_clusters: np.ndarray, min_cauda: int = 25):
    """
    Ajuste de Clauset, Shalizi & Newman (2009) em NumPy puro:
    Determina x_min que minimiza a estatística de Kolmogorov-Smirnov (KS)
    e estima gamma (lei de potência f(x) ~ x^-gamma) e alpha = 1 / (gamma - 1).
    """
    pop = np.asarray(pop_clusters, dtype=np.float64)
    pop = pop[pop > 0]
    if pop.size < max(10, min_cauda + 2):
        return dict(xmin=np.nan, n=int(pop.size), gamma=np.nan, erro_gamma=np.nan,
                    alpha=np.nan, erro_alpha=np.nan, ks=np.nan, alpha_ols=np.nan, r2_ols=np.nan)

    xs = np.sort(pop)
    n = xs.size
    lx = np.log(xs)
    soma_cauda = np.cumsum(lx[::-1])[::-1]
    _, inicio = np.unique(xs, return_index=True)

    corte = max(1, len(inicio) - min_cauda)
    melhor = None
    for k in inicio[:corte]:
        nt = n - k
        if nt < min_cauda:
            continue
        denom = soma_cauda[k] - nt * lx[k]
        if denom <= 0:
            continue
        gamma = 1.0 + nt / denom
        if gamma <= 1.05:
            continue
        cdf_emp = np.arange(1, nt + 1, dtype=np.float64) / nt
        cdf_teo = 1.0 - (xs[k:] / xs[k]) ** (1.0 - gamma)
        ks = float(np.max(np.abs(cdf_emp - cdf_teo)))
        if melhor is None or ks < melhor["ks"]:
            melhor = dict(xmin=float(xs[k]), n=int(nt), gamma=float(gamma), ks=ks)

    if melhor is None:
        return dict(xmin=float(xs[0]), n=int(n), gamma=np.nan, erro_gamma=np.nan,
                    alpha=np.nan, erro_alpha=np.nan, ks=np.nan, alpha_ols=np.nan, r2_ols=np.nan)

    gamma = melhor["gamma"]
    nx = melhor["n"]
    xmin = melhor["xmin"]
    melhor["erro_gamma"] = float((gamma - 1.0) / np.sqrt(nx))
    melhor["alpha"] = float(1.0 / (gamma - 1.0))
    melhor["erro_alpha"] = float(melhor["erro_gamma"] / ((gamma - 1.0) ** 2))

    xc = np.sort(pop[pop >= xmin])[::-1]
    rr = np.arange(1, xc.size + 1, dtype=np.float64)
    lx_r = np.log10(rr)
    ly_s = np.log10(xc)
    lx_c = lx_r - lx_r.mean()
    ly_c = ly_s - ly_s.mean()
    slope = float(np.sum(lx_c * ly_c) / np.sum(lx_c * lx_c))
    r_val = float(np.sum(lx_c * ly_c) / np.sqrt(np.sum(lx_c * lx_c) * np.sum(ly_c * ly_c)))
    melhor["alpha_ols"] = -slope
    melhor["r2_ols"] = r_val * r_val
    return melhor


@njit
def _raiz(pai, p):
    r = p
    while pai[r] != r:
        r = pai[r]
    while pai[p] != r:
        q = pai[p]
        pai[p] = r
        p = q
    return r


@njit
def _ativa_ate(D, pos, ordem, dens, popv, pai, tam, soma, ativo, di, dj, valido, nrows, ncols):
    while pos < ordem.size:
        p = ordem[pos]
        if dens[p] <= D:
            break
        ativo[p] = True
        pai[p] = p
        tam[p] = 1
        soma[p] = popv[p]
        r = p // ncols
        c = p - r * ncols
        for k in range(di.size):
            if valido[r, k]:
                rr = r + di[k]
                cc = c + dj[k]
                if 0 <= rr < nrows and 0 <= cc < ncols:
                    q = rr * ncols + cc
                    if ativo[q]:
                        a = _raiz(pai, p)
                        b = _raiz(pai, q)
                        if a != b:
                            if tam[a] < tam[b]:
                                tmp = a
                                a = b
                                b = tmp
                            pai[b] = a
                            tam[a] += tam[b]
                            soma[a] += soma[b]
        pos += 1
    return pos


@njit
def _pops_manchas(ordem, pos, pai, soma):
    out = np.empty(pos, dtype=np.float64)
    m = 0
    for i in range(pos):
        p = ordem[i]
        if pai[p] == p:
            out[m] = soma[p]
            m += 1
    return out[:m]


def calibrar_zipf_territorio(
    pop_count: np.ndarray,
    lat_centers_deg: np.ndarray,
    dlon_deg: float,
    dlat_deg: float,
    D_grade: np.ndarray = None,
    L_grade: np.ndarray = None,
    N_topo: tuple = (50, 100, 200),
):
    """
    Calibra (D*, l) para um território varrendo a grade D* x l e escolhendo o par com
    l >= l_min que minimiza Delta = max_N |alpha_N - 1| pela Lei de Zipf.
    """
    if D_grade is None:
        D_grade = np.arange(50, 2501, 50)
    if L_grade is None:
        L_grade = np.arange(500, 5001, 250)

    dens_km2, _ = calcular_densidade_gpw(pop_count, lat_centers_deg, dlon_deg, dlat_deg)
    nrows, ncols = dens_km2.shape
    dens_f = dens_km2.ravel()
    pop_f = np.nan_to_num(pop_count, nan=0.0).ravel()

    L_MIN = escala_minima_l_metros(lat_centers_deg, dlon_deg, dlat_deg)
    ordem = np.flatnonzero(dens_f > D_grade.min())
    ordem = ordem[np.argsort(-dens_f[ordem], kind="stable")].astype(np.int64)

    forma = (len(L_grade), len(D_grade))
    M = {f"alpha{n}": np.full(forma, np.nan) for n in N_topo}
    M["n_manchas"] = np.full(forma, np.nan)
    M["maior"] = np.full(forma, np.nan)

    pai = np.zeros(dens_f.size, dtype=np.int64)
    tam = np.zeros(dens_f.size, dtype=np.int64)
    soma = np.zeros(dens_f.size, dtype=np.float64)
    ativo = np.zeros(dens_f.size, dtype=np.bool_)

    chave_ant = None
    for i, l_m in enumerate(L_grade):
        di, dj, valido = vizinhanca_haversine(lat_centers_deg, dlon_deg, dlat_deg, l_m / 1000.0)
        chave = (di.tobytes(), dj.tobytes(), valido.tobytes())
        if chave == chave_ant:
            for c in M:
                M[c][i] = M[c][i - 1]
            continue
        chave_ant = chave
        ativo[ordem] = False
        pos = 0
        for j in range(len(D_grade) - 1, -1, -1):
            pos = _ativa_ate(
                float(D_grade[j]), pos, ordem, dens_f, pop_f,
                pai, tam, soma, ativo, di, dj, valido, nrows, ncols
            )
            x = _pops_manchas(ordem, pos, pai, soma)
            for n in N_topo:
                M[f"alpha{n}"][i, j] = alpha_hill(x, n)
            M["n_manchas"][i, j] = x.size
            M["maior"][i, j] = x.max() if x.size else np.nan

    n_validos = [n for n in N_topo if np.isfinite(M[f"alpha{n}"]).any()]
    if not n_validos:
        return dict(
            D_star=300, l_metros=max(L_MIN, 2000), L_MIN=L_MIN,
            alpha_hill={str(n): np.nan for n in N_topo}, delta=np.nan, matriz=M
        )

    A = np.stack([M[f"alpha{n}"] for n in n_validos])
    desvio = np.max(np.abs(A - 1.0), axis=0)
    permitido = (L_grade >= L_MIN)[:, None] & np.isfinite(desvio)
    if not permitido.any():
        permitido = np.isfinite(desvio)

    i_esc, j_esc = np.unravel_index(np.nanargmin(np.where(permitido, desvio, np.nan)), desvio.shape)
    D_esc = int(D_grade[j_esc])
    l_esc = int(L_grade[i_esc])
    alphas_esc = {str(n): float(M[f"alpha{n}"][i_esc, j_esc]) for n in N_topo}

    return dict(
        D_star=D_esc,
        l_metros=l_esc,
        L_MIN=L_MIN,
        alpha_hill=alphas_esc,
        delta=float(desvio[i_esc, j_esc]),
        n_manchas=int(M["n_manchas"][i_esc, j_esc]),
        maior_mancha_hab=float(M["maior"][i_esc, j_esc]),
        D_grade=D_grade,
        L_grade=L_grade,
        matriz=M,
    )


def alpha_ols_rank_tamanho(x: np.ndarray, n: int):
    """
    Regressão rank-tamanho log10(S_r) = log10(S_1) - alpha log10(r) nas N maiores manchas.
    Retorna (alpha, R²).
    """
    if x.size < n or n < 3:
        return np.nan, np.nan
    s = np.sort(np.partition(x, x.size - n)[-n:])[::-1]
    if s[-1] <= 0:
        return np.nan, np.nan
    lx = np.log10(np.arange(1, n + 1, dtype=np.float64))
    ly = np.log10(s)
    lxc = lx - lx.mean()
    lyc = ly - ly.mean()
    sxx = np.sum(lxc * lxc)
    syy = np.sum(lyc * lyc)
    if syy <= 0:
        return np.nan, np.nan
    sxy = np.sum(lxc * lyc)
    return float(-sxy / sxx), float(sxy * sxy / (sxx * syy))


def calibrar_zipf_dinamico(
    pop_count: np.ndarray,
    lat_centers_deg: np.ndarray,
    dlon_deg: float,
    dlat_deg: float,
    D_grade: np.ndarray = None,
    L_grade: np.ndarray = None,
    N_topo: tuple = (50, 100, 200),
    N_alvo: int = 100,
    r2_min: float = 0.98,
):
    """
    Calibração dinâmica (D*, L*) pela Lei de Zipf: para cada par da grade D x L roda o CCA
    (union-find incremental em D decrescente), ajusta a curva rank-tamanho por OLS nas N maiores
    manchas e escolhe o par com L >= L_min que minimiza |alpha_N - 1| sujeito a R²_N >= r2_min.
    Se nenhum par atinge r2_min, escolhe o de maior R² entre os de |alpha - 1| mínimo
    (flag `restricao_r2_atendida = False`).
    """
    if D_grade is None:
        D_grade = np.arange(100, 2001, 100)
    if L_grade is None:
        L_grade = np.arange(1000, 5001, 500)

    dens_km2, _ = calcular_densidade_gpw(pop_count, lat_centers_deg, dlon_deg, dlat_deg)
    nrows, ncols = dens_km2.shape
    dens_f = dens_km2.ravel()
    pop_f = np.nan_to_num(pop_count, nan=0.0).ravel()
    del dens_km2

    L_MIN = escala_minima_l_metros(lat_centers_deg, dlon_deg, dlat_deg)
    ordem = np.flatnonzero(dens_f > D_grade.min())
    ordem = ordem[np.argsort(-dens_f[ordem], kind="stable")].astype(np.int64)

    forma = (len(L_grade), len(D_grade))
    M = {}
    for n in N_topo:
        M[f"alpha{n}"] = np.full(forma, np.nan)
        M[f"alpha_ols{n}"] = np.full(forma, np.nan)
        M[f"r2_ols{n}"] = np.full(forma, np.nan)
    M["n_manchas"] = np.full(forma, np.nan)
    M["maior"] = np.full(forma, np.nan)

    pai = np.zeros(dens_f.size, dtype=np.int64)
    tam = np.zeros(dens_f.size, dtype=np.int64)
    soma = np.zeros(dens_f.size, dtype=np.float64)
    ativo = np.zeros(dens_f.size, dtype=np.bool_)

    chave_ant = None
    for i, l_m in enumerate(L_grade):
        di, dj, valido = vizinhanca_haversine(lat_centers_deg, dlon_deg, dlat_deg, l_m / 1000.0)
        chave = (di.tobytes(), dj.tobytes(), valido.tobytes())
        if chave == chave_ant:
            for c in M:
                M[c][i] = M[c][i - 1]
            continue
        chave_ant = chave
        ativo[ordem] = False
        pos = 0
        for j in range(len(D_grade) - 1, -1, -1):
            pos = _ativa_ate(
                float(D_grade[j]), pos, ordem, dens_f, pop_f,
                pai, tam, soma, ativo, di, dj, valido, nrows, ncols
            )
            x = _pops_manchas(ordem, pos, pai, soma)
            for n in N_topo:
                M[f"alpha{n}"][i, j] = alpha_hill(x, n)
                M[f"alpha_ols{n}"][i, j], M[f"r2_ols{n}"][i, j] = alpha_ols_rank_tamanho(x, n)
            M["n_manchas"][i, j] = x.size
            M["maior"][i, j] = x.max() if x.size else np.nan

    alpha = M[f"alpha_ols{N_alvo}"]
    r2 = M[f"r2_ols{N_alvo}"]
    desvio = np.abs(alpha - 1.0)
    fisico = (L_grade >= L_MIN)[:, None] & np.isfinite(desvio)
    if not fisico.any():
        fisico = np.isfinite(desvio)
    if not fisico.any():
        raise ValueError("Nenhum par (D, L) produziu manchas suficientes para o ajuste de Zipf.")

    permitido = fisico & (r2 >= r2_min)
    restricao_ok = bool(permitido.any())
    if restricao_ok:
        i_esc, j_esc = np.unravel_index(np.nanargmin(np.where(permitido, desvio, np.nan)), forma)
    else:
        d_min = np.nanmin(np.where(fisico, desvio, np.nan))
        empatados = fisico & np.isclose(desvio, d_min, atol=0.01)
        i_esc, j_esc = np.unravel_index(np.nanargmax(np.where(empatados, r2, np.nan)), forma)

    return dict(
        D_star=int(D_grade[j_esc]),
        l_metros=int(L_grade[i_esc]),
        L_MIN=L_MIN,
        N_alvo=int(N_alvo),
        r2_min=float(r2_min),
        restricao_r2_atendida=restricao_ok,
        alpha_ols=float(alpha[i_esc, j_esc]),
        r2_ols=float(r2[i_esc, j_esc]),
        alpha_ols_N={str(n): float(M[f"alpha_ols{n}"][i_esc, j_esc]) for n in N_topo},
        r2_ols_N={str(n): float(M[f"r2_ols{n}"][i_esc, j_esc]) for n in N_topo},
        alpha_hill={str(n): float(M[f"alpha{n}"][i_esc, j_esc]) for n in N_topo},
        delta=float(desvio[i_esc, j_esc]),
        n_manchas=int(M["n_manchas"][i_esc, j_esc]),
        maior_mancha_hab=float(M["maior"][i_esc, j_esc]),
        D_grade=D_grade,
        L_grade=L_grade,
        matriz=M,
    )


@njit
def _realocar_costa_bfs(bac_pop: np.ndarray, pop_positiva: np.ndarray, max_dist_px: int):
    """
    Realoca pixels habitados na linha de costa que caíram em pixels de oceano (bac_pop == 0)
    para a bacia terrestre mais próxima em até `max_dist_px` pixels.
    """
    nrows, ncols = bac_pop.shape
    out = bac_pop.copy()
    dist = np.full((nrows, ncols), -1, dtype=np.int16)

    q_r = np.empty(nrows * ncols, dtype=np.int32)
    q_c = np.empty(nrows * ncols, dtype=np.int32)
    head = 0
    tail = 0

    dr = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int32)
    dc = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int32)

    for r in range(nrows):
        for c in range(ncols):
            if bac_pop[r, c] > 0:
                dist[r, c] = 0
                tem_zero = False
                for d in range(8):
                    nr = r + dr[d]
                    nc = c + dc[d]
                    if 0 <= nr < nrows and 0 <= nc < ncols and bac_pop[nr, nc] == 0:
                        tem_zero = True
                        break
                if tem_zero:
                    q_r[tail] = r
                    q_c[tail] = c
                    tail += 1

    while head < tail:
        r = q_r[head]
        c = q_c[head]
        head += 1
        d_atual = dist[r, c]
        if d_atual >= max_dist_px:
            continue
        b_id = out[r, c]
        for d in range(8):
            nr = r + dr[d]
            nc = c + dc[d]
            if 0 <= nr < nrows and 0 <= nc < ncols:
                if dist[nr, nc] == -1:
                    dist[nr, nc] = d_atual + 1
                    out[nr, nc] = b_id
                    q_r[tail] = nr
                    q_c[tail] = nc
                    tail += 1

    res = bac_pop.copy()
    realocados = 0.0
    for r in range(nrows):
        for c in range(ncols):
            if bac_pop[r, c] == 0 and pop_positiva[r, c] > 0 and out[r, c] > 0:
                res[r, c] = out[r, c]
                realocados += pop_positiva[r, c]
    return res, realocados


def reamostrar_bacias_para_gpw(basin_grid: np.ndarray, shape_gpw: tuple) -> np.ndarray:
    """
    Alinha a grade de bacias (IPBA) à resolução da grade populacional GPW v4
    usando indexação por vizinho mais próximo em NumPy puro.
    """
    h_b, w_b = basin_grid.shape
    h_p, w_p = shape_gpw
    if (h_b, w_b) == (h_p, w_p):
        return basin_grid.astype(np.int32)
    rows = np.clip((np.arange(h_p) * h_b // h_p), 0, h_b - 1)
    cols = np.clip((np.arange(w_p) * w_b // w_p), 0, w_b - 1)
    return basin_grid[rows[:, None], cols[None, :]].astype(np.int32)


def dividir_manchas_e_somar_populacao_bacias(
    cidades_grid: np.ndarray,
    basin_grid_dem: np.ndarray,
    pop_count: np.ndarray,
    row_areas_gpw_km2: np.ndarray,
    major_basin_ids: np.ndarray,
    max_realoc_px: int = 8,
):
    """
    Realiza a contagem de população por bacia e a divisão de manchas urbanas entre bacias
    somando diretamente a CONTAGEM DE PESSOAS POR PIXEL (`pop_count` do GPW v4) dentro de cada bacia.
    """
    pop_clean = np.where(np.isfinite(pop_count) & (pop_count > 0), pop_count, 0.0).astype(np.float64)
    bac_gpw = reamostrar_bacias_para_gpw(basin_grid_dem, pop_clean.shape)
    bac_pop, hab_realocados = _realocar_costa_bfs(bac_gpw, pop_clean, max_realoc_px)

    max_b = int(max(bac_pop.max(), major_basin_ids.max() if major_basin_ids.size else 0)) + 1
    flat_bac = bac_pop.ravel()
    flat_pop = pop_clean.ravel()
    flat_cid = cidades_grid.ravel()
    urb_mask = flat_cid > 0

    soma_total_por_id = np.bincount(flat_bac, weights=flat_pop, minlength=max_b)
    soma_urb_por_id = np.bincount(flat_bac[urb_mask], weights=flat_pop[urb_mask], minlength=max_b)

    pop_total = soma_total_por_id[major_basin_ids]
    pop_urbana = soma_urb_por_id[major_basin_ids]

    area_2d = np.broadcast_to(row_areas_gpw_km2[:, None], pop_clean.shape).ravel()
    cid_u = flat_cid[urb_mask].astype(np.int64)
    bac_u = flat_bac[urb_mask].astype(np.int64)
    pop_u = flat_pop[urb_mask]
    area_u = area_2d[urb_mask]

    if cid_u.size > 0:
        fator = int(bac_u.max()) + 1
        chave_par = cid_u * fator + bac_u
        chaves_unicas, inv = np.unique(chave_par, return_inverse=True)
        mancha_par = (chaves_unicas // fator).astype(np.int32)
        bacia_par = (chaves_unicas % fator).astype(np.int32)
        pop_par = np.bincount(inv, weights=pop_u)
        area_par = np.bincount(inv, weights=area_u)

        max_c = int(mancha_par.max()) + 1
        pop_mancha_tot = np.bincount(mancha_par, weights=pop_par, minlength=max_c)
        fracao_par = pop_par / np.maximum(1e-12, pop_mancha_tot[mancha_par])
    else:
        mancha_par = np.zeros(0, dtype=np.int32)
        bacia_par = np.zeros(0, dtype=np.int32)
        pop_par = np.zeros(0, dtype=np.float64)
        area_par = np.zeros(0, dtype=np.float64)
        fracao_par = np.zeros(0, dtype=np.float64)

    pedacos_dict = dict(
        mancha=mancha_par,
        basin_id=bacia_par,
        pop=pop_par,
        area_km2=area_par,
        fracao_pop=fracao_par,
        bacia_principal=np.isin(bacia_par, major_basin_ids),
        hab_realocados=float(hab_realocados),
        hab_fora_bacias=float(flat_pop[flat_bac == 0].sum()),
    )
    return bac_pop, pop_total, pop_urbana, pedacos_dict

