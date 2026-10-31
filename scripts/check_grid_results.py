import pandas as pd

df = pd.read_csv("relatorios/grid_search_limpeza_nao_industrial.csv")
print("Total linhas:", len(df))

print("\n--- TOP 15 CONFIGURAÇÕES NÃO-INDUSTRIAIS COM N >= 10 BACIAS (BETA > 0) ---")
sub = df[(df["N"] >= 10) & (df["beta"] > 0)].sort_values(by="r2", ascending=False)
for idx, r in sub.head(15).iterrows():
    sig = "***" if r["p_val"] < 0.01 else ("**" if r["p_val"] < 0.05 else ("*" if r["p_val"] < 0.1 else ""))
    print(f"R²={r['r2']*100:5.1f}% | r={r['r']:+5.2f} (p={r['p_val']:.4f} {sig:3s}) | beta={r['beta']:+5.2f}±{r['se']:.2f} | N={r['N']:2d} | rho={r['rho']:+5.2f} | R_cap={r['raio_cap_km']:2.0f}km N>={r['n_min_samples']} outl={r['sem_outliers']} | {r['subconjunto'][:25]} | {r['pop_col']} vs {r['met_col']}")

print("\n--- TOP 15 CONFIGURAÇÕES NÃO-INDUSTRIAIS GERAIS (QUALQUER N >= 4, BETA > 0) ---")
sub_all = df[df["beta"] > 0].sort_values(by="r2", ascending=False)
for idx, r in sub_all.head(15).iterrows():
    sig = "***" if r["p_val"] < 0.01 else ("**" if r["p_val"] < 0.05 else ("*" if r["p_val"] < 0.1 else ""))
    print(f"R²={r['r2']*100:5.1f}% | r={r['r']:+5.2f} (p={r['p_val']:.4f} {sig:3s}) | beta={r['beta']:+5.2f}±{r['se']:.2f} | N={r['N']:2d} | rho={r['rho']:+5.2f} | R_cap={r['raio_cap_km']:2.0f}km N>={r['n_min_samples']} outl={r['sem_outliers']} | {r['subconjunto'][:25]} | {r['pop_col']} vs {r['met_col']}")
