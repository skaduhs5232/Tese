"""
Lei de Zipf das manchas urbanas do Brasil e do Nordeste (GPW v4 2020 + CCA), para o artigo.

Usa a mesma varredura (D*, l) por união e busca, o mesmo estimador de Hill e o mesmo ajuste de
Clauset-Shalizi-Newman do notebooks/01_Zipf_calibracao_CCA.ipynb. O CCA roda sobre o Brasil
inteiro; uma mancha é do Nordeste quando a maior parte da sua população está nos 9 estados da
região, para que as divisas estaduais não cortem manchas. A comparação administrativa usa a
população dos municípios no Censo 2022 (IBGE, SIDRA tabela 4709).

Entradas (baixadas uma vez):
  dados/gpw_data/*.tif                      GPW v4.11, contagem 2020, 30"
  dados/ibge/brasil.geojson                 malha do país (API de malhas do IBGE)
  dados/ibge/nordeste_ufs.geojson           malha das 9 UFs do Nordeste
  dados/ibge/pop_municipios_censo2022.json  SIDRA t/4709, v/93
Saídas: artigo/resultados_brasil/
"""

import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import powerlaw
import rioxarray
from numba import njit
from rasterio.features import rasterize
from scipy import ndimage as ndi
from scipy import stats
from shapely.geometry import shape

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DADOS = os.path.join(RAIZ, "dados")
SAIDA = os.path.join(AQUI, "resultados_brasil")
os.makedirs(SAIDA, exist_ok=True)

BOX = (-74.1, -33.8, -28.8, 5.3)       # Brasil continental, Fernando de Noronha e Trindade
D_GRADE = np.arange(10, 3001, 10)       # hab/km²
L_GRADE = np.arange(100, 5001, 100)     # metros
N_TOPO = (50, 100, 200)
R_TERRA = 6378.137                      # km, o mesmo raio do pacote cca
CENSUS = (390, 3000)                    # 386 hab/km² do U.S. Census na grade de 10 hab/km²

REFS = {
    # Nordeste
    "Salvador": (-38.50, -12.97), "Fortaleza": (-38.54, -3.73), "Recife": (-34.88, -8.05),
    "São Luís": (-44.30, -2.53), "Natal": (-35.21, -5.79), "Maceió": (-35.73, -9.67),
    "João Pessoa": (-34.86, -7.12), "Teresina": (-42.80, -5.09), "Aracaju": (-37.07, -10.91),
    "Feira de Santana": (-38.97, -12.27), "Campina Grande": (-35.88, -7.23),
    "Juazeiro do Norte": (-39.31, -7.21), "Caruaru": (-35.98, -8.28), "Petrolina": (-40.50, -9.39),
    "Mossoró": (-37.34, -5.19), "Vitória da Conquista": (-40.84, -14.86), "Imperatriz": (-47.49, -5.53),
    "Itabuna": (-39.28, -14.79), "Sobral": (-40.35, -3.69), "Arapiraca": (-36.66, -9.75),
    "Camaçari": (-38.32, -12.70), "Parnaíba": (-41.78, -2.90),
    # resto do país
    "São Paulo": (-46.63, -23.55), "Rio de Janeiro": (-43.20, -22.91), "Belo Horizonte": (-43.94, -19.92),
    "Brasília": (-47.88, -15.79), "Porto Alegre": (-51.23, -30.03), "Curitiba": (-49.27, -25.43),
    "Goiânia": (-49.25, -16.68), "Belém": (-48.50, -1.46), "Manaus": (-60.02, -3.12),
    "Campinas": (-47.06, -22.91), "Vitória": (-40.31, -20.32), "Florianópolis": (-48.55, -27.60),
    "Santos": (-46.33, -23.96),
}


# ------------------------------------------------------------------ dados
def carrega():
    gpw = rioxarray.open_rasterio(glob.glob(os.path.join(DADOS, "gpw_data", "**", "*.tif"), recursive=True)[0])
    pop = gpw.squeeze().rio.clip_box(*BOX)
    pop = pop.where(pop > 0, 0.0)
    lon, lat = pop.x.values, pop.y.values
    transf = pop.rio.transform()
    forma = pop.shape

    br = json.load(open(os.path.join(DADOS, "ibge", "brasil.geojson"), encoding="utf-8"))
    ne = json.load(open(os.path.join(DADOS, "ibge", "nordeste_ufs.geojson"), encoding="utf-8"))
    no_br = rasterize([shape(f["geometry"]) for f in br["features"]], out_shape=forma, transform=transf,
                      all_touched=True, fill=0, default_value=1, dtype="uint8").astype(bool)
    uf = rasterize([(shape(f["geometry"]), int(f["properties"]["codarea"])) for f in ne["features"]],
                   out_shape=forma, transform=transf, all_touched=True, fill=0, dtype="uint8")
    no_ne = (uf > 0) & no_br

    pop_px = np.where(no_br, np.nan_to_num(pop.values, nan=0.0), 0.0).astype(np.float64)
    dx, dy = np.abs(pop.rio.resolution())
    area_px_km2 = ((dx * 111.32) * np.cos(np.radians(lat)))[:, None] * (dy * 111.32)
    return pop_px, no_ne, lon, lat, dx, dy, area_px_km2


def municipios():
    d = json.load(open(os.path.join(DADOS, "ibge", "pop_municipios_censo2022.json"), encoding="utf-8"))[1:]
    m = pd.DataFrame({"cod": [r["D1C"] for r in d], "nome": [r["D1N"] for r in d],
                      "pop": pd.to_numeric([r["V"] for r in d], errors="coerce")}).dropna()
    m["ne"] = m["cod"].str[0] == "2"
    return m


# ------------------------------------------------------------------ CCA por união e busca
def vizinhanca(l_km, lat, dx, dy):
    """Deslocamentos (di, dj) candidatos e, por linha do raster, se o par fica a menos de l (haversine)."""
    dlat, dlon = np.radians(dy), np.radians(dx)
    cos_min = np.cos(np.radians(np.abs(lat).max() + dy))
    ri = int(np.ceil(l_km / (R_TERRA * dlat)))
    rj = int(np.ceil(l_km / (R_TERRA * dlon * cos_min)))
    di, dj = np.meshgrid(np.arange(-ri, ri + 1), np.arange(-rj, rj + 1), indexing="ij")
    fora = (di == 0) & (dj == 0)
    di, dj = di[~fora], dj[~fora]
    passo_lat = np.radians(lat[1] - lat[0])
    lat0 = np.radians(lat)[:, None]
    lat1 = lat0 + di[None, :] * passo_lat
    h = np.sin((lat1 - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dj[None, :] * dlon / 2) ** 2
    valido = 2 * R_TERRA * np.arcsin(np.sqrt(np.clip(h, 0, 1))) < l_km
    usados = valido.any(axis=0)
    return di[usados].astype(np.int64), dj[usados].astype(np.int64), np.ascontiguousarray(valido[:, usados])


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
def ativa_ate(D, pos, ordem, dens, popv, popne, pai, tam, soma, soma_ne, ativo, di, dj, valido, nrows, ncols):
    """Ativa (em ordem decrescente) os pixels com densidade > D e une cada um aos vizinhos ativos a menos de l."""
    n = ordem.size
    while pos < n:
        p = ordem[pos]
        if dens[p] <= D:
            break
        ativo[p] = True
        pai[p] = p
        tam[p] = 1
        soma[p] = popv[p]
        soma_ne[p] = popne[p]
        r = p // ncols
        c = p - r * ncols
        for k in range(di.size):
            if not valido[r, k]:
                continue
            rr = r + di[k]
            cc = c + dj[k]
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                continue
            q = rr * ncols + cc
            if ativo[q]:
                a = _raiz(pai, p)
                b = _raiz(pai, q)
                if a != b:
                    if tam[a] < tam[b]:
                        a, b = b, a
                    pai[b] = a
                    tam[a] += tam[b]
                    soma[a] += soma[b]
                    soma_ne[a] += soma_ne[b]
        pos += 1
    return pos


@njit
def pops_manchas(ordem, pos, pai, soma, soma_ne):
    """Populações de todas as manchas e das manchas com maioria da população no Nordeste."""
    todas = np.empty(pos)
    ne = np.empty(pos)
    m = k = 0
    for i in range(pos):
        p = ordem[i]
        if pai[p] == p:
            todas[m] = soma[p]
            m += 1
            if soma_ne[p] > 0.5 * soma[p]:
                ne[k] = soma[p]
                k += 1
    return todas[:m], ne[:k]


@njit
def raizes(ordem, pos, pai):
    out = np.empty(pos, dtype=np.int64)
    for i in range(pos):
        out[i] = _raiz(pai, ordem[i])
    return out


def alpha_hill(x, n):
    """Inclinação do rank-size das n maiores manchas: alpha = 1/zeta, zeta de Hill = (n-1) / sum ln(S_i/S_n)."""
    if x.size < n:
        return np.nan
    s = np.partition(x, x.size - n)[-n:]
    return float(np.sum(np.log(s / s.min())) / (n - 1))


class Varredura:
    def __init__(self, pop_px, no_ne, lat, dx, dy, area_px_km2):
        self.nrows, self.ncols = pop_px.shape
        self.lat, self.dx, self.dy = lat, dx, dy
        self.dens = (pop_px / area_px_km2).ravel()
        self.pop = pop_px.ravel()
        self.popne = np.where(no_ne, pop_px, 0.0).ravel()
        ordem = np.flatnonzero(self.dens > D_GRADE.min())
        self.ordem = ordem[np.argsort(-self.dens[ordem], kind="stable")].astype(np.int64)
        n = self.dens.size
        self.pai = np.zeros(n, dtype=np.int64)
        self.tam = np.zeros(n, dtype=np.int64)
        self.soma = np.zeros(n)
        self.soma_ne = np.zeros(n)
        self.ativo = np.zeros(n, dtype=np.bool_)

    def passada(self, l_m, D_lista):
        """Uma passada para um l: devolve, para cada D* de D_lista (decrescente), as manchas (todas, NE)."""
        di, dj, valido = vizinhanca(l_m / 1000, self.lat, self.dx, self.dy)
        self.ativo[self.ordem] = False
        pos = 0
        for D in D_lista:
            pos = ativa_ate(float(D), pos, self.ordem, self.dens, self.pop, self.popne, self.pai, self.tam,
                            self.soma, self.soma_ne, self.ativo, di, dj, valido, self.nrows, self.ncols)
            yield D, pos, pops_manchas(self.ordem, pos, self.pai, self.soma, self.soma_ne)

    def rotulos(self, D, l_m):
        """Raster de rótulos do CCA com (D, l): 0 fora das manchas, 1..n nas manchas."""
        for _, pos, _ in self.passada(l_m, [D]):
            pass
        r = raizes(self.ordem, pos, self.pai)
        _, lab = np.unique(r, return_inverse=True)
        out = np.zeros(self.dens.size, dtype=np.int64)
        out[self.ordem[:pos]] = lab + 1
        return out.reshape(self.nrows, self.ncols)


# ------------------------------------------------------------------ ajuste CSN (igual ao notebook 01)
def _melhor_xmin(x, min_cauda=30):
    xs = np.sort(np.asarray(x, dtype=float))
    n = len(xs)
    lx = np.log(xs)
    soma_cauda = np.cumsum(lx[::-1])[::-1]
    _, inicio = np.unique(xs, return_index=True)
    melhor = None
    for k in inicio[:-min_cauda]:
        nt = n - k
        gamma = 1.0 + nt / (soma_cauda[k] - nt * lx[k])
        if gamma <= 1.05:
            continue
        cdf_teo = 1.0 - (xs[k:] / xs[k]) ** (1.0 - gamma)
        ks = float(np.max(np.abs(np.arange(1, nt + 1) / nt - cdf_teo)))
        if melhor is None or ks < melhor["ks"]:
            melhor = dict(xmin=float(xs[k]), n=int(nt), gamma=float(gamma), ks=ks)
    return melhor


def ajusta_zipf(pop, n_boot=200, semente=42):
    pop = np.asarray(pop, dtype=float)
    pop = pop[pop > 0]
    melhor = _melhor_xmin(pop)
    gamma, nx, xmin = melhor["gamma"], melhor["n"], melhor["xmin"]
    melhor["erro_gamma"] = (gamma - 1.0) / np.sqrt(nx)
    melhor["alpha"] = 1.0 / (gamma - 1.0)
    melhor["erro_alpha"] = melhor["erro_gamma"] / (gamma - 1.0) ** 2
    melhor["ks_critico"] = 1.36 / np.sqrt(nx)
    rng = np.random.default_rng(semente)
    corpo = pop[pop < xmin]
    n = len(pop)
    piores = validos = 0
    for _ in range(n_boot):
        k = rng.binomial(n, nx / n)
        cauda = xmin * (1.0 - rng.random(k)) ** (-1.0 / (gamma - 1.0))
        sint = np.concatenate([cauda, rng.choice(corpo, n - k)]) if len(corpo) else cauda
        rs = _melhor_xmin(sint)
        if rs is None:
            continue
        validos += 1
        piores += rs["ks"] >= melhor["ks"]
    melhor["p_valor"] = (piores / validos) if validos else float("nan")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        f = powerlaw.Fit(pop, xmin=xmin, verbose=False)
        for alt in ("lognormal", "exponential", "truncated_power_law"):
            R, p = f.distribution_compare("power_law", alt, normalized_ratio=True)
            melhor[f"R_{alt}"], melhor[f"p_{alt}"] = float(R), float(p)
    melhor["alpha_hill"] = {str(n): alpha_hill(pop, n) for n in N_TOPO}
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in melhor.items()}


def escolhe(M, l_min):
    A = np.stack([M[f"alpha{n}"] for n in N_TOPO])
    desvio = np.max(np.abs(A - 1), axis=0)
    desvio[L_GRADE < l_min] = np.nan
    i, j = np.unravel_index(np.nanargmin(desvio), desvio.shape)
    return int(D_GRADE[j]), int(L_GRADE[i]), desvio


def tabela_manchas(lab, pop_px, no_ne, lon, lat, area_px_km2):
    n = int(lab.max())
    idx = np.arange(1, n + 1)
    pop = ndi.sum(pop_px, lab, idx)
    pop_ne = ndi.sum(np.where(no_ne, pop_px, 0.0), lab, idx)
    area = ndi.sum(area_px_km2 * np.ones_like(pop_px), lab, idx)
    cy, cx = np.array(ndi.center_of_mass(pop_px, lab, idx)).T
    df = pd.DataFrame({"mancha": idx, "pop": pop, "area_km2": area, "ne": pop_ne > 0.5 * pop,
                       "lon": np.interp(cx, np.arange(lon.size), lon), "lat": np.interp(cy, np.arange(lat.size), lat)})
    nomes = {}
    for nome, (lo, la) in REFS.items():
        r, c = int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())
        jan = lab[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4]
        pj = pop_px[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4]
        if (jan > 0).any():
            ids, inv = np.unique(jan[jan > 0], return_inverse=True)
            nomes.setdefault(int(ids[np.bincount(inv, weights=pj[jan > 0]).argmax()]), nome)
    df["nome"] = df["mancha"].map(nomes)
    return df.sort_values("pop", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ principal
def main():
    t0 = time.time()
    pop_px, no_ne, lon, lat, dx, dy, area_px_km2 = carrega()
    print(f"GPW: {pop_px.shape} | Brasil {pop_px.sum():,.0f} hab | Nordeste {pop_px[no_ne].sum():,.0f} hab")

    i_eq = int(np.abs(lat).argmin())
    la0, la1 = np.radians(lat[i_eq]), np.radians(lat[i_eq] + dy)
    diag_km = 2 * R_TERRA * np.arcsin(np.sqrt(np.sin((la1 - la0) / 2) ** 2
                                              + np.cos(la0) * np.cos(la1) * np.sin(np.radians(dx) / 2) ** 2))
    L_MIN = int(np.ceil(diag_km * 10) * 100)

    V = Varredura(pop_px, no_ne, lat, dx, dy, area_px_km2)
    print(f"{V.ordem.size:,} pixels com densidade > {D_GRADE.min()} hab/km² | l_min = {L_MIN} m")

    # conferência: com l = 1,4 km o CCA no Brasil (|lat| <= 34°) é a vizinhança de 8 pixels
    lab8, n8 = ndi.label(pop_px / area_px_km2 > 1000, structure=np.ones((3, 3)))
    lab_v = V.rotulos(1000, 1400)
    assert n8 == lab_v.max(), (n8, lab_v.max())
    print(f"conferência com ndimage.label (D*=1000, l=1,4 km): {n8} manchas nas duas")

    forma = (len(L_GRADE), len(D_GRADE))
    chaves = [f"alpha{n}" for n in N_TOPO] + ["n_manchas", "maior"]
    M = {r: {c: np.full(forma, np.nan) for c in chaves} for r in ("br", "ne")}
    for i, l_m in enumerate(L_GRADE):
        for D, _, (todas, ne) in V.passada(l_m, D_GRADE[::-1]):
            j = int(np.flatnonzero(D_GRADE == D)[0])
            for r, x in (("br", todas), ("ne", ne)):
                for n in N_TOPO:
                    M[r][f"alpha{n}"][i, j] = alpha_hill(x, n)
                M[r]["n_manchas"][i, j] = x.size
                M[r]["maior"][i, j] = x.max() if x.size else np.nan
        print(f"l = {l_m:>4} m | {time.time() - t0:,.0f} s")
    for r in M:
        np.savez_compressed(os.path.join(SAIDA, f"zipf_matriz_{r}.npz"), D_star=D_GRADE, l_metros=L_GRADE,
                            L_MIN=L_MIN, **M[r])

    mun = municipios()
    resumo = {"box": BOX, "l_min_m": L_MIN, "pop_brasil": float(pop_px.sum()), "pop_nordeste": float(pop_px[no_ne].sum()),
              "grade": {"D_star": [10, 3000, 10], "l_metros": [100, 5000, 100]}, "N": list(N_TOPO)}
    for r in ("br", "ne"):
        D_esc, l_esc, desvio = escolhe(M[r], L_MIN)
        i, j = int(np.flatnonzero(L_GRADE == l_esc)[0]), int(np.flatnonzero(D_GRADE == D_esc)[0])
        ic, jc = int(np.flatnonzero(L_GRADE == CENSUS[1])[0]), int(np.flatnonzero(D_GRADE == CENSUS[0])[0])
        lab = V.rotulos(D_esc, l_esc)
        df = tabela_manchas(lab, pop_px, no_ne, lon, lat, area_px_km2)
        df.to_csv(os.path.join(SAIDA, f"manchas_{r}.csv"), index=False)
        x = df.loc[df["ne"], "pop"].to_numpy() if r == "ne" else df["pop"].to_numpy()
        perto = np.argwhere(desvio < 0.10)
        resumo[r] = {
            "D_star": D_esc, "l_metros": l_esc, "delta": float(desvio[i, j]),
            "alpha_hill": {str(n): float(M[r][f"alpha{n}"][i, j]) for n in N_TOPO},
            "n_manchas": int(x.size), "pop_manchas": float(x.sum()),
            "pares_delta_menor_0.10": int(perto.shape[0]),
            "faixa_delta_menor_0.10": ({"D_star": [int(D_GRADE[perto[:, 1]].min()), int(D_GRADE[perto[:, 1]].max())],
                                        "l_metros": [int(L_GRADE[perto[:, 0]].min()), int(L_GRADE[perto[:, 0]].max())]}
                                       if perto.size else None),
            "census": {"alpha_hill": {str(n): float(M[r][f"alpha{n}"][ic, jc]) for n in N_TOPO},
                       "delta": float(np.max([abs(M[r][f"alpha{n}"][ic, jc] - 1) for n in N_TOPO]))},
            "faixa_alpha100_l_ge_lmin": [float(np.nanmin(M[r]["alpha100"][L_GRADE >= L_MIN])),
                                         float(np.nanmax(M[r]["alpha100"][L_GRADE >= L_MIN]))],
            "csn_manchas": ajusta_zipf(x),
            "csn_municipios": ajusta_zipf(mun.loc[mun["ne"], "pop"] if r == "ne" else mun["pop"]),
            "n_municipios": int(mun["ne"].sum() if r == "ne" else len(mun)),
        }
        print(r, json.dumps({k: v for k, v in resumo[r].items() if not k.startswith("csn")}, ensure_ascii=False))
        print(r, "CSN manchas", {k: round(v, 3) for k, v in resumo[r]["csn_manchas"].items() if isinstance(v, float)})
        print(r, "CSN municípios", {k: round(v, 3) for k, v in resumo[r]["csn_municipios"].items() if isinstance(v, float)})
    mun.to_csv(os.path.join(SAIDA, "municipios_censo2022.csv"), index=False)
    with open(os.path.join(SAIDA, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    complementa()
    print(f"pronto em {time.time() - t0:,.0f} s")


def complementa():
    """Nordeste com a definição nacional de cidade: manchas do par calibrado para o Brasil com maioria
    da população no Nordeste, e o Nordeste nas matrizes nesse par."""
    caminho = os.path.join(SAIDA, "resumo.json")
    resumo = json.load(open(caminho, encoding="utf-8"))
    D_br, l_br = resumo["br"]["D_star"], resumo["br"]["l_metros"]
    b = pd.read_csv(os.path.join(SAIDA, "manchas_br.csv"))
    ne = b[b["ne"]]
    M = np.load(os.path.join(SAIDA, "zipf_matriz_ne.npz"))
    i, j = int(np.flatnonzero(M["l_metros"] == l_br)[0]), int(np.flatnonzero(M["D_star"] == D_br)[0])
    ok = M["l_metros"] >= resumo["l_min_m"]
    resumo["ne_par_br"] = {
        "D_star": D_br, "l_metros": l_br, "n_manchas": int(len(ne)), "pop_manchas": float(ne["pop"].sum()),
        "alpha_hill": {str(n): float(M[f"alpha{n}"][i, j]) for n in N_TOPO},
        "alpha50_mediana_l_ge_lmin": float(np.nanmedian(M["alpha50"][ok])),
        "alpha50_faixa_l_ge_lmin": [float(np.nanmin(M["alpha50"][ok])), float(np.nanmax(M["alpha50"][ok]))],
        "maiores": ne.head(15)[["nome", "pop", "area_km2"]].to_dict("records"),
        "csn_manchas": ajusta_zipf(ne["pop"].to_numpy()),
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print("ne_par_br", {k: v for k, v in resumo["ne_par_br"].items() if k not in ("maiores", "csn_manchas")})


if __name__ == "__main__":
    import sys
    complementa() if "--complemento" in sys.argv else main()
