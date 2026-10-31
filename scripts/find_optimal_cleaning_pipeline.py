"""
scripts/find_optimal_cleaning_pipeline.py
-----------------------------------------
Varredura de estratégias de limpeza para maximizar a correlação estatística
e o R² entre População da Bacia e Microplásticos Não-Industriais.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.spatial_matcher import match_sinks_with_individual_radius

def carregar_dados_limpos():
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
        method = str(p.get("Sampling_Method", "")).strip()
        val_mp = p.get("Microplastics_measurement")
        ref = str(p.get("Short_Reference", "")).strip()

        # Filtrar exclusivamente NÃO-INDUSTRIAIS com valor numérico positivo
        if med in ["Ocean water", "Beach", "Ocean sediment"] and val_mp is not None:
            v = float(val_mp)
            if v > 0:
                records.append({
                    "latitude": lat,
                    "longitude": lon,
                    "measurement": v,
                    "nurdles_per_10min": v, # compatibilidade com match_sinks
                    "medium": med,
                    "method": method,
                    "short_ref": ref,
                })

    df_mp = pd.DataFrame(records)
    sinks_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "sinks_raios_north_central_america.csv")
    pop_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "populacao_todas_bacias_north_central_america.csv")

    return df_mp, sinks_na, pop_na


def testar_configuracoes():
    df_mp, sinks_na, pop_na = carregar_dados_limpos()
    print(f"Total registros Não-Industriais carregados: {len(df_mp)}")

    # Testar diferentes subconjuntos de métodos
    subconjuntos = {
        "1. Todos Não-Industriais (Água + Praia)": df_mp,
        "2. Redes de Superfície (Neuston + Manta)": df_mp[df_mp["method"].isin(["Neuston net", "Manta net"])],
        "3. Apenas Ocean Water": df_mp[df_mp["medium"] == "Ocean water"],
        "4. Apenas Beach (Praia)": df_mp[df_mp["medium"] == "Beach"],
        "5. Sem Van Dorn Sampler (Remover garrafas de litro com microfibras)": df_mp[df_mp["method"] != "Van Dorn sampler"],
        "6. Redes de Superfície sem Outliers (>100 pcs/m3)": df_mp[df_mp["method"].isin(["Neuston net", "Manta net"]) & (df_mp["measurement"] <= 100.0)],
    }

    # Testar diferentes caps de raio máximo (15 km, 25 km, 50 km)
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

            # Filtros de Esforço Amostral
            for n_min in [1, 2, 3]:
                df_f = df_merged[df_merged["n_samples"] >= n_min].copy()
                if len(df_f) < 4:
                    continue

                # Filtro de Outliers em log10 da medição (z <= 2.5)
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
    print("TOP 20 CONFIGURAÇÕES COM MAIOR R² (CORRELAÇÃO POSITIVA SIGNIFICANTE):")
    print("=" * 85)
    # Filtrar correlações positivas com p_val razoável
    pos_res = df_res[df_res["beta"] > 0].sort_values(by="r2", ascending=False)
    for idx, r in pos_res.head(20).iterrows():
        sig = "***" if r["p_val"] < 0.01 else ("**" if r["p_val"] < 0.05 else ("*" if r["p_val"] < 0.1 else ""))
        print(f"R²={r['r2']*100:5.1f}% | r={r['r']:+5.2f} (p={r['p_val']:.4f} {sig:3s}) | beta={r['beta']:+5.2f}±{r['se']:.2f} | N={r['N']:2d} | rho={r['rho']:+5.2f} | R_cap={r['raio_cap_km']:2.0f}km N>={r['n_min_samples']} outl={r['sem_outliers']} | {r['subconjunto'][:25]} | {r['pop_col']} vs {r['met_col']}")

    return df_res

if __name__ == "__main__":
    testar_configuracoes()
