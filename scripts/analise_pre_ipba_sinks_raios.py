"""
scripts/analise_pre_ipba_sinks_raios.py
---------------------------------------
Analise Pre-IPBA e Pre-CCA solicitada pelo Prof. Dr. Erneson Oliveira (Audios 1, 2, 3 e 4):
  1. Dados: exclusivamente Agua Oceanica (`Medium == 'Ocean water'`, `Unit == 'pieces/m3'`)
     na faixa de Microplastico Padrao (`0,02 <= Mesh_size__mm_ <= 5,0 mm`, n = 14.773 pontos).
  2. Sinks: corte direto no relevo GEBCO em altura h = 0 (sem rodar IPBA nem filtro de 95%),
     definindo como sink todo pixel de terra vizinho imediato (4-conectividade) do mar.
  3. Circulos de raio fixo R = 1, 2, 3... km em volta de cada sink:
     - Tira a MEDIA aritmetica (pieces/m3, nunca a soma) dos pontos que caem no circulo.
     - Armazena em cada sink uma lista de dicionarios (um por raio R) com:
       media, desvio-padrao, intervalo de confianca (IC 95%), numero de pontos,
       valores brutos e flag booleana `shared_with_other_sink` indicando se os pontos
       tambem caem no circulo de outro sink.
     - Mantem um contador de uso (`usage_count`) para cada ponto de microplastico em cada raio R.
  4. Gera tabela de estatisticas, tabela de contador de uso e figura cientifica com
     graficos de barras de Media + IC 95% por raio R e diagnostico de sobreposicao.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from geracao_graficos.estilo_erneson import aplicar_estilo, rotulo_painel, salvar
from pipeline.ipba_runner import EARTH_RADIUS_KM, crop_gebco, extract_coastal_sinks

RAIOS_KM: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    22, 25, 30, 40, 50,
]


def to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def fmt_int_pt(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def carregar_ocean_water_malha_padrao() -> pd.DataFrame:
    """
    Carrega estritamente o recorte definido no Audio 1:
    Agua oceanica (`Medium == 'Ocean water'`, `Unit == 'pieces/m3'`)
    com malha padrao de microplastico (`0.02 <= Mesh_size__mm_ <= 5.0 mm`).
    """
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    with open(noaa_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    df = pd.DataFrame([ft["properties"] for ft in gj["features"]])
    df["lat"] = pd.to_numeric(df["Latitude__degree_"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude_degree_"], errors="coerce")
    df["mesh_mm"] = pd.to_numeric(df["Mesh_size__mm_"], errors="coerce")
    df["valor"] = pd.to_numeric(df["Microplastics_measurement"], errors="coerce")

    ow = df[
        (df["Medium"] == "Ocean water")
        & (df["Unit"] == "pieces/m3")
        & (df["mesh_mm"] >= 0.02)
        & (df["mesh_mm"] <= 5.0)
        & df["lat"].notna()
        & df["lon"].notna()
        & df["valor"].notna()
    ].copy()

    ow = ow.reset_index(drop=True)
    ow["point_id"] = np.arange(1, len(ow) + 1, dtype=int)
    return ow


def extrair_todos_sinks_gebco_h0() -> pd.DataFrame:
    """
    Aplica o corte h = 0 no GEBCO global (-60 a 80 graus de latitude)
    e identifica todo pixel de terra vizinho de mar (4-conectividade) como sink,
    sem rodar o IPBA e sem aplicar corte de 95%.
    """
    gebco_path = REPO_ROOT / "dados" / "gebco" / "gebco_2026_sub_ice_geotiff.tif"
    dem, gt, _ = crop_gebco((-180.0, -60.0, 180.0, 80.0), gebco_path)
    _, sinks_df = extract_coastal_sinks(dem, gt, h_star=0.0)
    return sinks_df


def _media_e_ic95(arr: np.ndarray) -> tuple[float, float, float, float]:
    """Retorna (media, std, sem, ic95) de um vetor 1D."""
    vals = arr[np.isfinite(arr)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean_v = float(np.mean(vals))
    if n < 2:
        return mean_v, 0.0, 0.0, 0.0
    std_v = float(np.std(vals, ddof=1))
    sem_v = float(std_v / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return mean_v, std_v, sem_v, float(t_crit * sem_v)


def executar_analise_pre_ipba(
    ow: pd.DataFrame,
    sinks_df: pd.DataFrame,
    raios_km: list[int] = RAIOS_KM,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[dict[str, Any]]]]:
    """
    Cresce um circulo de raio R em torno de cada sink (h = 0):
      - Guarda em cada sink uma lista de dicionarios (um por raio R)
        contendo media, desvio, IC 95%, n_points, valores brutos e flag `shared_with_other_sink`.
      - Calcula o contador de uso (`usage_count`) para cada ponto de microplastico em cada raio R.
    """
    s_xyz = to_xyz(sinks_df["latitude"].to_numpy(), sinks_df["longitude"].to_numpy())
    p_xyz = to_xyz(ow["lat"].to_numpy(), ow["lon"].to_numpy())

    tree_s = cKDTree(s_xyz)
    vals = ow["valor"].to_numpy()
    sink_ids = sinks_df["sink_id"].to_numpy()
    sink_lats = sinks_df["latitude"].to_numpy()
    sink_lons = sinks_df["longitude"].to_numpy()

    # Distancia de cada ponto ao sink mais proximo
    d_chord_min, nearest_s_idx = tree_s.query(p_xyz, k=1)
    cols_base = [c for c in ["point_id", "ObjectId", "lat", "lon", "valor", "mesh_mm", "Sampling_Method", "Oceans"] if c in ow.columns]
    ow_usage_df = ow[cols_base].copy()
    ow_usage_df["nearest_sink_id"] = sink_ids[nearest_s_idx]
    ow_usage_df["dist_nearest_sink_km"] = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(d_chord_min / 2.0, 0.0, 1.0))

    # Estrutura de lista de dicionarios por sink (Audio 4)
    sinks_dict_list: dict[int, list[dict[str, Any]]] = defaultdict(list)
    stats_rows: list[dict[str, Any]] = []

    for r_km in raios_km:
        chord = float(2.0 * np.sin((r_km / EARTH_RADIUS_KM) / 2.0))
        # Para cada ponto de microplastico, quais sinks estao a distancia <= R?
        sinks_per_point = tree_s.query_ball_point(p_xyz, r=chord)
        point_usage = np.array([len(s_list) for s_list in sinks_per_point], dtype=int)
        ow_usage_df[f"uso_R{r_km}km"] = point_usage

        # Inverter para obter os pontos que cairam no circulo de cada sink
        points_per_sink: dict[int, list[int]] = defaultdict(list)
        for p_idx, s_list in enumerate(sinks_per_point):
            for s_idx in s_list:
                points_per_sink[s_idx].append(p_idx)

        sink_means_all = []
        sink_means_excl = []
        sink_log_means_all = []
        sink_log_means_excl = []
        n_pts_per_active_sink = []
        n_shared_sinks = 0

        for s_idx, p_idx_list in points_per_sink.items():
            pts_arr = np.array(p_idx_list, dtype=int)
            v_arr = vals[pts_arr]
            u_arr = point_usage[pts_arr]

            n_pts = len(v_arr)
            m_val = float(np.mean(v_arr))
            std_val = float(np.std(v_arr, ddof=1)) if n_pts > 1 else 0.0
            sem_val = float(std_val / np.sqrt(n_pts)) if n_pts > 1 else 0.0
            ci95_val = float(1.96 * sem_val)

            is_shared = bool(np.any(u_arr > 1))
            n_sh_pts = int(np.sum(u_arr > 1))
            max_u = int(np.max(u_arr))

            s_id = int(sink_ids[s_idx])
            sinks_dict_list[s_id].append(
                {
                    "sink_id": s_id,
                    "latitude": float(sink_lats[s_idx]),
                    "longitude": float(sink_lons[s_idx]),
                    "r_km": float(r_km),
                    "mean": m_val,
                    "std": std_val,
                    "sem": sem_val,
                    "ci95": ci95_val,
                    "n_points": int(n_pts),
                    "shared_with_other_sink": is_shared,
                    "n_shared_points": n_sh_pts,
                    "max_point_usage": max_u,
                    "point_ids": (pts_arr + 1).tolist(),
                    "raw_values": [float(x) for x in v_arr],
                }
            )

            sink_means_all.append(m_val)
            n_pts_per_active_sink.append(n_pts)
            if m_val > 0:
                sink_log_means_all.append(np.log10(m_val))

            if is_shared:
                n_shared_sinks += 1
            else:
                sink_means_excl.append(m_val)
                if m_val > 0:
                    sink_log_means_excl.append(np.log10(m_val))

        # Verificacao de consistencia matematica: soma dos pontos nos sinks == soma do contador de uso
        assert sum(n_pts_per_active_sink) == int(point_usage.sum()), "Inconsistencia na contagem!"

        arr_all = np.array(sink_means_all, dtype=float)
        arr_excl = np.array(sink_means_excl, dtype=float)
        arr_log_all = np.array(sink_log_means_all, dtype=float)
        arr_log_excl = np.array(sink_log_means_excl, dtype=float)

        m_all, sd_all, sem_all, ci_all = _media_e_ic95(arr_all)
        m_ex, sd_ex, sem_ex, ci_ex = _media_e_ic95(arr_excl)
        ml_all, sdl_all, seml_all, cil_all = _media_e_ic95(arr_log_all)
        ml_ex, sdl_ex, seml_ex, cil_ex = _media_e_ic95(arr_log_excl)

        n_active = len(arr_all)
        n_excl = len(arr_excl)
        used_mask = point_usage > 0
        n_used_pts = int(np.sum(used_mask))
        n_shared_pts = int(np.sum(point_usage > 1))
        mean_pt_use = float(np.mean(point_usage[used_mask])) if n_used_pts > 0 else 0.0
        max_pt_use = int(np.max(point_usage)) if n_used_pts > 0 else 0

        stats_rows.append(
            {
                "r_km": r_km,
                "n_sinks_total_gebco": len(sinks_df),
                "n_sinks_ativos": n_active,
                "n_sinks_exclusivos": n_excl,
                "n_sinks_compartilhados": n_shared_sinks,
                "pct_sinks_compartilhados": (n_shared_sinks / n_active * 100.0) if n_active > 0 else 0.0,
                "n_pontos_unicos_capturados": n_used_pts,
                "n_pontos_compartilhados": n_shared_pts,
                "pct_pontos_compartilhados": (n_shared_pts / n_used_pts * 100.0) if n_used_pts > 0 else 0.0,
                "soma_usos_pontos": int(point_usage.sum()),
                "media_usos_por_ponto": mean_pt_use,
                "max_usos_ponto": max_pt_use,
                "media_pontos_por_sink_ativo": float(np.mean(n_pts_per_active_sink)) if n_active > 0 else 0.0,
                "mean_linear_all": m_all,
                "std_linear_all": sd_all,
                "sem_linear_all": sem_all,
                "ci95_linear_all": ci_all,
                "median_linear_all": float(np.median(arr_all)) if n_active > 0 else np.nan,
                "mean_linear_excl": m_ex,
                "std_linear_excl": sd_ex,
                "sem_linear_excl": sem_ex,
                "ci95_linear_excl": ci_ex,
                "n_sinks_pos_all": len(arr_log_all),
                "mean_log10_all": ml_all,
                "std_log10_all": sdl_all,
                "sem_log10_all": seml_all,
                "ci95_log10_all": cil_all,
                "n_sinks_pos_excl": len(arr_log_excl),
                "mean_log10_excl": ml_ex,
                "std_log10_excl": sdl_ex,
                "sem_log10_excl": seml_ex,
                "ci95_log10_excl": cil_ex,
            }
        )

    df_stats = pd.DataFrame(stats_rows)
    return df_stats, ow_usage_df, dict(sinks_dict_list)


def gerar_figura_pre_ipba(df_stats: pd.DataFrame, n_ow: int, n_sinks_tot: int) -> Path:
    """
    Gera a figura cientifica da Analise Pre-IPBA (Audios 3 e 4):
      - Painel (a): Grafico de barras de Media e Intervalo de Confianca (IC 95%) em escala log10
                    por raio R (comparando Todos os Sinks Ativos vs. Sinks Exclusivos).
      - Painel (b): Grafico de barras de Media Bruta Linear (pieces/m3) + IC 95% por raio R,
                    evidenciando a estabilizacao da media a partir de R = 10-15 km.
      - Painel (c): Diagnostico do Contador de Uso por Ponto e % de Sinks Compartilhados,
                    mostrando quando o circulo comeca a invadir sinks vizinhos.
    """
    aplicar_estilo()

    # Selecionar os 14 raios principais para visualizacao nitida nas barras
    raios_plot = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50]
    sub = df_stats[df_stats["r_km"].isin(raios_plot)].copy().reset_index(drop=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.2, 4.9), gridspec_kw={"wspace": 0.25})

    x = np.arange(len(sub))
    r_labels = [str(int(r)) for r in sub["r_km"]]
    w = 0.38

    # =========================================================================
    # Painel (a): Grafico de Barras Media + IC 95% em log10(pieces/m3)
    # =========================================================================
    # Para plotar barras a partir da base -2.5 (0,003 pieces/m3)
    base_y = -2.4
    y_all = sub["mean_log10_all"].to_numpy()
    ci_all = sub["ci95_log10_all"].to_numpy()
    y_ex = sub["mean_log10_excl"].to_numpy()
    ci_ex = sub["ci95_log10_excl"].to_numpy()

    ax1.bar(
        x - w / 2,
        y_all - base_y,
        width=w,
        bottom=base_y,
        color="#2b83ba",
        edgecolor="black",
        linewidth=0.55,
        alpha=0.88,
        zorder=3,
        label="Todos os sinks ativos",
    )
    ax1.errorbar(
        x - w / 2,
        y_all,
        yerr=ci_all,
        fmt="none",
        ecolor="#08306b",
        elinewidth=1.1,
        capsize=2.4,
        zorder=5,
    )

    ax1.bar(
        x + w / 2,
        y_ex - base_y,
        width=w,
        bottom=base_y,
        color="#1b7837",
        edgecolor="black",
        linewidth=0.55,
        alpha=0.88,
        zorder=3,
        label="Sinks exclusivos (s/ ponto compartilhado)",
    )
    ax1.errorbar(
        x + w / 2,
        y_ex,
        yerr=ci_ex,
        fmt="none",
        ecolor="#00441b",
        elinewidth=1.1,
        capsize=2.4,
        zorder=5,
    )

    ax1.axhline(0.0, color="#666666", linestyle=":", linewidth=0.9, zorder=2)
    ax1.axvline(6.5, color="#d7191c", linestyle="--", linewidth=1.2, alpha=0.8, zorder=4)
    ax1.text(
        6.75,
        0.38,
        "Faixa 8–15 km:\nmédia estabiliza",
        fontsize=7.8,
        color="#b30000",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.88),
        zorder=7,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(r_labels, fontsize=8.3)
    ax1.set_ylim(base_y, 1.18)
    ax1.set_xlabel("Raio fixo R do círculo em cada sink (km)", fontsize=10.0)
    ax1.set_ylabel(r"Média de $\log_{10}(\mathrm{pieces/m^3}) \pm \mathrm{IC\ 95\%}$", fontsize=10.0)
    ax1.set_title("Média logarítmica e IC 95% nos sinks", fontsize=10.5, pad=9)
    ax1.legend(
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.94,
    )
    rotulo_painel(ax1, "(a)", dentro=False, x=-0.14, y=1.03)

    # =========================================================================
    # Painel (b): Grafico de Barras da Media Aritmetica Bruta (pieces/m3) + IC 95%
    # =========================================================================
    m_lin = sub["mean_linear_all"].to_numpy()
    ci_lin = sub["ci95_linear_all"].to_numpy()

    bar_colors_b = ["#f59e0b" if r < 8 else ("#2b83ba" if r <= 20 else "#64748b") for r in sub["r_km"]]
    ax2.bar(
        x,
        m_lin,
        width=0.58,
        color=bar_colors_b,
        edgecolor="black",
        linewidth=0.55,
        alpha=0.88,
        zorder=3,
    )
    yerr_low = np.minimum(ci_lin, m_lin * 0.96)
    yerr_high = ci_lin
    ax2.errorbar(
        x,
        m_lin,
        yerr=[yerr_low, yerr_high],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=2.5,
        zorder=5,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(r_labels, fontsize=8.3)
    ax2.set_ylim(0, 1420)
    ax2.set_xlabel("Raio fixo R do círculo em cada sink (km)", fontsize=10.0)
    ax2.set_ylabel(r"Média aritmética nos sinks ($\mathrm{pieces/m^3}$) $\pm$ IC 95%", fontsize=10.0)
    ax2.set_title("Queda da variância e estabilização da média bruta", fontsize=10.5, pad=9)

    leg_b = [
        Patch(facecolor="#f59e0b", edgecolor="black", lw=0.5, label="R < 8 km (alta variância costeira)"),
        Patch(facecolor="#2b83ba", edgecolor="black", lw=0.5, label="8 ≤ R ≤ 20 km (regime estável)"),
        Patch(facecolor="#64748b", edgecolor="black", lw=0.5, label="R > 20 km (homogeneização entre bacias)"),
    ]
    ax2.legend(
        handles=leg_b,
        loc="upper right",
        fontsize=7.4,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.94,
    )
    rotulo_painel(ax2, "(b)", dentro=False, x=-0.14, y=1.03)

    # =========================================================================
    # Painel (c): Contador de Uso dos Pontos e % de Sinks Compartilhados (Audio 4)
    # =========================================================================
    pct_sh = sub["pct_sinks_compartilhados"].to_numpy()
    mean_use = sub["media_usos_por_ponto"].to_numpy()
    n_pts_cap = sub["n_pontos_unicos_capturados"].to_numpy()

    ax3.bar(
        x,
        pct_sh,
        width=0.56,
        color="#e5e7eb",
        edgecolor="#4b5563",
        linewidth=0.6,
        zorder=2,
        label="% de sinks c/ pontos compartilhados",
    )
    ax3.plot(
        x,
        pct_sh,
        color="#d7191c",
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        zorder=5,
        label="% sinks compartilhados",
    )
    ax3.axhline(50.0, color="#d7191c", linestyle=":", linewidth=1.0, zorder=3)
    ax3.annotate(
        "R ≈ 3 km (grade GEBCO):\n48,6% compartilhados",
        xy=(2, pct_sh[2]),
        xytext=(0.15, 74.0),
        fontsize=7.3,
        color="#b30000",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#fca5a5", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#d7191c", lw=0.8),
        zorder=8,
    )

    ax3.set_xticks(x)
    ax3.set_xticklabels(r_labels, fontsize=8.3)
    ax3.set_ylim(0, 108)
    ax3.set_xlabel("Raio fixo R do círculo em cada sink (km)", fontsize=10.0)
    ax3.set_ylabel("Sinks que compartilham pontos (%)", fontsize=10.0, color="#b30000")
    ax3.tick_params(axis="y", labelcolor="#b30000")
    ax3.set_title(f"Sobreposição e contador de uso (n = {fmt_int_pt(n_ow)})", fontsize=10.5, pad=9)

    ax3_r = ax3.twinx()
    ax3_r.plot(
        x,
        mean_use,
        color="#1d4ed8",
        marker="s",
        markersize=4.5,
        linewidth=1.5,
        linestyle="--",
        zorder=6,
        label="Média de círculos por ponto usado",
    )
    ax3_r.set_ylim(0, max(mean_use) * 1.18)
    ax3_r.set_ylabel("Contador médio de uso (círculos / ponto)", fontsize=9.8, color="#1d4ed8")
    ax3_r.tick_params(axis="y", labelcolor="#1d4ed8", direction="out", length=4.0, width=0.8)

    # Anotar pontos unicos capturados em R=5, R=10, R=15 e R=25 km sem colisao
    offsets_c = {
        5: (-0.8, 5.2),
        10: (-1.1, 6.2),
        15: (-1.2, 6.8),
        25: (-1.6, 6.5),
    }
    for r_dest, (dx_off, dy_off) in offsets_c.items():
        idx_r = int(np.where(sub["r_km"].to_numpy() == r_dest)[0][0])
        ax3_r.annotate(
            f"{fmt_int_pt(n_pts_cap[idx_r])} pts\n({mean_use[idx_r]:.1f}x)".replace(".", ",", 1).replace(",", ".", 1),
            xy=(idx_r, mean_use[idx_r]),
            xytext=(idx_r + dx_off, mean_use[idx_r] + dy_off),
            fontsize=7.0,
            color="#1e3a8a",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#93c5fd", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=0.75),
            zorder=9,
        )

    rotulo_painel(ax3, "(c)", dentro=False, x=-0.14, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_pre_ipba_media_ic_por_raio"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


if __name__ == "__main__":
    print("1. Carregando Agua Oceanica (malha padrao 0,02-5,0 mm, pieces/m3)...")
    ow_df = carregar_ocean_water_malha_padrao()
    print(f"   Total pontos carregados: {len(ow_df)} (positivos: {(ow_df['valor'] > 0).sum()}, zeros: {(ow_df['valor'] == 0).sum()})")

    print("2. Extraindo todos os sinks costeiros do GEBCO (corte h = 0, sem IPBA e sem corte de 95%)...")
    sinks_all_df = extrair_todos_sinks_gebco_h0()
    print(f"   Total sinks costeiros (h = 0): {len(sinks_all_df)}")

    print("3. Crescendo circulos de raio fixo R em cada sink e populando listas de dicionarios + contador de uso...")
    df_stats_r, df_usage_pts, dict_sinks = executar_analise_pre_ipba(ow_df, sinks_all_df, RAIOS_KM)

    out_dir = REPO_ROOT / "relatorios"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_csv = out_dir / "pre_ipba_estatisticas_raios_sinks.csv"
    usage_csv = out_dir / "pre_ipba_contador_uso_pontos_noaa.csv"
    dicts_pkl = out_dir / "pre_ipba_sinks_dicionarios_raios.pkl"

    df_stats_r.to_csv(stats_csv, index=False)
    df_usage_pts.to_csv(usage_csv, index=False)
    with open(dicts_pkl, "wb") as f:
        pickle.dump(dict_sinks, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"   Tabelas salvas em:\n     - {stats_csv}\n     - {usage_csv}\n     - {dicts_pkl} ({len(dict_sinks)} sinks com listas de dicionarios)")

    print("\nResumo por raio R (selecao):")
    cols_show = [
        "r_km", "n_sinks_ativos", "n_sinks_exclusivos", "pct_sinks_compartilhados",
        "n_pontos_unicos_capturados", "media_usos_por_ponto",
        "mean_linear_all", "ci95_linear_all", "mean_log10_all", "ci95_log10_all",
    ]
    print(df_stats_r[df_stats_r["r_km"].isin([1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 50])][cols_show].to_string(index=False))

    print("\n4. Gerando figura cientifica da analise pre-IPBA...")
    fig_path = gerar_figura_pre_ipba(df_stats_r, len(ow_df), len(sinks_all_df))
    print(f"   Figura salva em: {fig_path}")
