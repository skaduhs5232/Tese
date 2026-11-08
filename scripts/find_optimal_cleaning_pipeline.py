"""
scripts/find_optimal_cleaning_pipeline.py
-----------------------------------------
Varredura de estratégias de limpeza na base NOAA NCEI restrita estritamente
a MICROPLÁSTICOS VERIFICADOS POR MALHA (< 5 mm, unidade volumétrica `pieces/m3`),
além da verificação numérica da fórmula de Triângulos Esféricos (Teorema de L'Huilier
+ Haversine, Equações 3 e 4 de Oliveira et al., 2019).
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.ipba_runner import EARTH_RADIUS_KM, spherical_triangle_row_areas
from pipeline.spatial_matcher import match_sinks_with_individual_radius


def verificar_ganho_triangulos_esfericos() -> pd.DataFrame:
    """
    Compara a área de sítios (pixels) calculada pela composição de dois triângulos
    esféricos (Teorema de L'Huilier + Haversine, Oliveira et al., 2019) contra a
    aproximação diferencial R² cos(lat) dlon dlat, a integral zonal esférica e o
    cálculo ingênuo sem correção latitudinal (ex.: Rasterio em graus fixos).
    """
    casos = [
        ("Baixa Latitude (Equador, 0° - pixel GPW 30 arc-sec)", 0.0, 1.0 / 120.0),
        ("Média Latitude (45° - pixel GPW 30 arc-sec)", 45.0, 1.0 / 120.0),
        ("Alta Latitude (70° - pixel GPW 30 arc-sec)", 70.0, 1.0 / 120.0),
        ("Baixa Latitude (Equador, 0° - pixel GEBCO 15 arc-sec)", 0.0, 1.0 / 240.0),
        ("Alta Latitude (70° - pixel GEBCO 15 arc-sec)", 70.0, 1.0 / 240.0),
    ]
    registros = []
    for desc, lat_c, res_deg in casos:
        lats_arr = np.array([lat_c], dtype=float)
        a_lhuilier = float(spherical_triangle_row_areas(lats_arr, res_deg, res_deg, R=EARTH_RADIUS_KM)[0])
        dlon_r = np.radians(res_deg)
        dlat_r = np.radians(res_deg)
        a_diff = float((EARTH_RADIUS_KM ** 2) * np.cos(np.radians(lat_c)) * dlon_r * dlat_r)
        lat1_r = np.radians(lat_c - 0.5 * res_deg)
        lat2_r = np.radians(lat_c + 0.5 * res_deg)
        a_zonal = float((EARTH_RADIUS_KM ** 2) * (np.sin(lat2_r) - np.sin(lat1_r)) * dlon_r)
        a_naive = float(((res_deg * 111.31949) ** 2))

        registros.append({
            "cenario": desc,
            "area_spherical_triangles_km2": a_lhuilier,
            "area_zonal_exact_km2": a_zonal,
            "area_diff_approx_km2": a_diff,
            "area_naive_no_lat_km2": a_naive,
            "diff_lhuilier_vs_approx_pct": abs(a_lhuilier - a_diff) / a_lhuilier * 100.0,
            "erro_naive_sem_lat_pct": abs(a_naive - a_lhuilier) / a_lhuilier * 100.0,
        })
    return pd.DataFrame(registros)


def carregar_dados_limpos(apenas_malha_padrao: bool = False):
    """
    Carrega exclusivamente registros de MICROPLÁSTICOS VERIFICADOS (< 5 mm) da NOAA NCEI:
      - Exclui pellets industriais pré-produção (`Nurdle Patrol`, `pieces/10 mins`).
      - Exclui `Ocean sediment` em massa seca (`pieces kg-1 d.w.`), mantendo apenas
        concentração volumétrica homogênea (`Unit == 'pieces/m3'`).
      - Exige abertura de malha/peneira documentada na faixa de microplásticos:
        `0 < Mesh_size__mm_ <= 5.0` (excluindo `Hand picking` a olho nu sem malha).
      - Se `apenas_malha_padrao=True`, restringe a `0.02 <= Mesh_size__mm_ <= 5.0 mm`
        (20 um a 5 mm), eliminando amostradores pontuais de garrafa/bomba de 1 litro
        com filtros de membrana sub-micrométricos (< 20 um: Van Dorn, Niskin, Grab sample, Pump).
    """
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    with open(noaa_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    records = []
    for feat in gj.get("features", []):
        p = feat.get("properties") or {}
        lat = p.get("Latitude__degree_")
        lon = p.get("Longitude_degree_")
        if lat is None or lon is None:
            continue
        lat, lon = float(lat), float(lon)

        med = str(p.get("Medium", "")).strip()
        unit = str(p.get("Unit", "")).strip()
        method = str(p.get("Sampling_Method", "")).strip()
        val_mp = p.get("Microplastics_measurement")
        mesh_raw = p.get("Mesh_size__mm_")
        ref = str(p.get("Short_Reference", "")).strip()

        if med in ["Ocean water", "Beach"] and unit == "pieces/m3" and val_mp is not None and mesh_raw is not None:
            try:
                v = float(val_mp)
                mesh_mm = float(mesh_raw)
            except (ValueError, TypeError):
                continue

            min_mesh = 0.02 if apenas_malha_padrao else 1e-6
            if v > 0 and (min_mesh <= mesh_mm <= 5.0):
                records.append({
                    "latitude": lat,
                    "longitude": lon,
                    "measurement": v,
                    "nurdles_per_10min": v,  # compatibilidade com match_sinks
                    "medium": med,
                    "unit": unit,
                    "method": method,
                    "mesh_mm": mesh_mm,
                    "short_ref": ref,
                })

    df_mp = pd.DataFrame(records)
    sinks_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "sinks_raios_north_central_america.csv")
    pop_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "populacao_todas_bacias_north_central_america.csv")

    return df_mp, sinks_na, pop_na


def testar_configuracoes():
    df_bench = verificar_ganho_triangulos_esfericos()
    print("=" * 85)
    print("VERIFICAÇÃO DE ÁREA: TRIÂNGULOS ESFÉRICOS (L'HUILIER) VS APROXIMAÇÃO E RASTERIO")
    print("=" * 85)
    for _, row in df_bench.iterrows():
        print(
            f"{row['cenario'][:48]:48s} | L'Huilier={row['area_spherical_triangles_km2']:.6f} km² "
            f"| Diff vs Aprox={row['diff_lhuilier_vs_approx_pct']:.7f}% "
            f"| Erro Sem Lat={row['erro_naive_sem_lat_pct']:.2f}%"
        )

    df_mp, sinks_na, pop_na = carregar_dados_limpos(apenas_malha_padrao=False)
    print(f"\nTotal registros de Microplásticos Verificados (< 5 mm, pieces/m³) carregados: {len(df_mp)}")

    # Subconjuntos estritos de microplásticos (< 5 mm, pieces/m³)
    subconjuntos = {
        "1. Microplásticos Malha Padrão (0.02 a 5.0 mm, pieces/m³)": df_mp[
            (df_mp["mesh_mm"] >= 0.02) & (df_mp["mesh_mm"] <= 5.0)
        ],
        "2. Redes Superficiais de Microplásticos (Neuston/Manta/Plankton/Bongo/AVANI)": df_mp[
            (df_mp["medium"] == "Ocean water")
            & (df_mp["method"].isin(["Neuston net", "Manta net", "Plankton net", "Bongo net", "AVANI net"]))
        ],
        "3. Microplásticos Peneirados (Mesh <= 5.0 mm, sem Van Dorn)": df_mp[
            df_mp["method"] != "Van Dorn sampler"
        ],
        "4. Apenas Ocean Water Malha Padrão (0.02 a 5.0 mm)": df_mp[
            (df_mp["medium"] == "Ocean water") & (df_mp["mesh_mm"] >= 0.02) & (df_mp["mesh_mm"] <= 5.0)
        ],
        "5. Apenas Beach Peneirado (0.038 a 1.13 mm, sem Hand picking)": df_mp[
            (df_mp["medium"] == "Beach") & (df_mp["mesh_mm"] >= 0.02) & (df_mp["mesh_mm"] <= 5.0)
        ],
    }

    radii_caps = [15.0, 25.0, 50.0]
    resultados = []

    for nome_sub, df_sub in subconjuntos.items():
        if len(df_sub) < 10:
            continue

        for r_cap in radii_caps:
            matched_mp, summary, _ = match_sinks_with_individual_radius(sinks_na, df_sub, max_cap_km=r_cap)
            if len(summary) < 4:
                continue

            df_merged = summary.merge(pop_na, on="basin_id", how="inner")

            for n_min in [1, 2, 3]:
                df_f = df_merged[df_merged["n_samples"] >= n_min].copy()
                if len(df_f) < 4:
                    continue

                for remove_outliers in [False, True]:
                    if remove_outliers:
                        if len(df_f) < 5:
                            continue
                        log_y = np.log10(df_f["nurdles_median"].values)
                        z = np.abs((log_y - np.nanmean(log_y)) / (np.nanstd(log_y) + 1e-9))
                        df_curr = df_f[z <= 2.5].copy()
                    else:
                        df_curr = df_f.copy()

                    if len(df_curr) < 4:
                        continue

                    for pop_col in ["total_population", "urban_population"]:
                        for met_col in ["nurdles_median", "nurdles_mean", "nurdles_sum"]:
                            mask = (df_curr[pop_col] > 0) & (df_curr[met_col] > 0)
                            xs = df_curr.loc[mask, pop_col].values
                            ys = df_curr.loc[mask, met_col].values
                            n = len(xs)
                            if n < 4:
                                continue

                            lx = np.log10(xs)
                            ly = np.log10(ys)

                            slope, intercept, r_val, p_val, se = stats.linregress(lx, ly)
                            rho, p_rho = stats.spearmanr(xs, ys)

                            resultados.append({
                                "subconjunto": nome_sub,
                                "raio_cap_km": r_cap,
                                "n_min_samples": n_min,
                                "sem_outliers": remove_outliers,
                                "pop_col": pop_col,
                                "met_col": met_col,
                                "N": n,
                                "beta": slope,
                                "se": se,
                                "r2": r_val ** 2,
                                "r": r_val,
                                "p_val": p_val,
                                "rho": rho,
                                "p_rho": p_rho,
                            })

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(REPO_ROOT / "relatorios" / "grid_search_limpeza_nao_industrial.csv", index=False)
    print(f"Total combinações testadas: {len(df_res)}")

    print("\n" + "=" * 85)
    print("TOP 20 CONFIGURAÇÕES COM N >= 10 BACIAS E MAIOR R² (MICROPLÁSTICOS < 5 mm):")
    print("=" * 85)
    pos_res = df_res[(df_res["beta"] > 0) & (df_res["N"] >= 10)].sort_values(by="r2", ascending=False)
    for idx, r in pos_res.head(20).iterrows():
        sig = "***" if r["p_val"] < 0.01 else ("**" if r["p_val"] < 0.05 else ("*" if r["p_val"] < 0.1 else ""))
        print(
            f"R²={r['r2']*100:5.1f}% | r={r['r']:+5.2f} (p={r['p_val']:.4f} {sig:3s}) "
            f"| beta={r['beta']:+5.2f}±{r['se']:.2f} | N={r['N']:2d} | rho={r['rho']:+5.2f} "
            f"| R_cap={r['raio_cap_km']:2.0f}km N>={r['n_min_samples']} outl={r['sem_outliers']} "
            f"| {r['subconjunto'][:28]} | {r['pop_col']} vs {r['met_col']}"
        )

    return df_res


if __name__ == "__main__":
    testar_configuracoes()
