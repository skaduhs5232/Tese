"""
Gera versão detalhada, lenta e didática do GIF de conurbação de Fortaleza (CCA):
c:\\faculdade\\mestrado\\Tese\\artigo\\figs\\gifs\\04_cca_fortaleza_conurbacao_metropolitana.gif

Evolui suavemente ao longo de ~36 segundos (em ritmo cadenciado), permitindo acompanhar com clareza:
1. Relevo demográfico contínuo (GPW v4) com núcleos destacados (3,5 s)
2. Núcleos urbanos isolados nos centros municipais (D* decrescendo lentamente)
3. 1ª Conurbação: Caucaia conecta-se a Fortaleza pela faixa costeira/BR-222 (pausa de destaque de 3,0 s)
4. 2ª Conurbação: Maracanaú conecta-se pelo corredor industrial sul (pausa de destaque de 3,0 s)
5. Expansão periurbana contígua e consolidação dos subúrbios
6. Mancha consolidada final da Figura (a) da tese (D* = 1.450 hab/km², l = 1,8 km): 3,35M hab, 370 km² (pausa de 6,0 s)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
if AQUI not in sys.path:
    sys.path.insert(0, AQUI)

import analise_brasil as ab

GIFS_DIR = os.path.join(AQUI, "figs", "gifs")
os.makedirs(GIFS_DIR, exist_ok=True)
OUT_PATH = os.path.join(GIFS_DIR, "04_cca_fortaleza_conurbacao_metropolitana.gif")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Latin Modern Roman"],
    "mathtext.fontset": "cm",
})


def gerar_gif_04_fortaleza_lento(out_path=None):
    if out_path is None:
        out_path = OUT_PATH
        
    print("\n[4/5] Gerando GIF 4: Conurbação Metropolitana de Fortaleza (Versão Lenta & Didática)...")
    print("Carregando raster GPW v4 e preparando recorte de Fortaleza...")
    pop_px, no_ne, lon, lat, dx, dy, area_px_km2 = ab.carrega()
    dens_grid = pop_px / area_px_km2
    area_grid_km2 = np.broadcast_to(area_px_km2, pop_px.shape)

    # Bounding box refinado de Fortaleza Metropolitana
    # lat: [-4.02, -3.62], lon: [-38.78, -38.36]
    r_min = int(np.abs(lat - (-3.62)).argmin())
    r_max = int(np.abs(lat - (-4.02)).argmin())
    c_min = int(np.abs(lon - (-38.78)).argmin())
    c_max = int(np.abs(lon - (-38.36)).argmin())

    sub_dens = dens_grid[r_min:r_max, c_min:c_max]
    sub_pop = pop_px[r_min:r_max, c_min:c_max]
    sub_area = area_grid_km2[r_min:r_max, c_min:c_max]
    sub_lat = lat[r_min:r_max]
    sub_lon = lon[c_min:c_max]
    extent = [sub_lon[0], sub_lon[-1], sub_lat[-1], sub_lat[0]]

    # Células de pico das 3 cidades para identificação robusta de clusters
    rf_peak = int(np.abs(sub_lat - (-3.7125)).argmin())
    cf_peak = int(np.abs(sub_lon - (-38.5542)).argmin())

    rc_peak = int(np.abs(sub_lat - (-3.7708)).argmin())
    cc_peak = int(np.abs(sub_lon - (-38.6542)).argmin())

    rm_peak = int(np.abs(sub_lat - (-3.8792)).argmin())
    cm_peak = int(np.abs(sub_lon - (-38.6125)).argmin())

    # Pontos de referência para rótulos no mapa
    PONTOS_FOR = {
        "Fortaleza (Centro / Aldeota)": (-38.528, -3.715),
        "Caucaia": (-38.654, -3.771),
        "Maracanaú": (-38.613, -3.879),
        "Messejana": (-38.498, -3.832),
        "Eusébio": (-38.455, -3.890),
    }

    V = ab.Varredura(pop_px, no_ne, lat, dx, dy, area_px_km2)
    lab_calib = V.rotulos(1450, 1800)
    id_fortaleza_calib = lab_calib[r_min + rf_peak, c_min + cf_peak]

    # ------------------------------------------------------------------ Cronograma Cuidadosamente Cadenciado
    # Cada entrada: (tipo_fase, D_val, duracao_ms, texto_status, cor_destaque)
    cronograma = []

    # 1. Introdução: Densidade Contínua (3,5 s de pausa para observação do relevo térmico)
    cronograma.append(("intro_densidade", None, 3500, "Apresentação: Relevo Demográfico Contínuo (GPW v4, 2020)", "#0284c7"))

    # 2. Fase 1: Nucleação e Núcleos Isolados (descida lenta e gradual de D*)
    d_nucleos = [
        (16000, 800, "Fase 1: Núcleos pioneiros de hiperdensidade (> 16.000 hab/km²)", "#475569"),
        (14000, 750, "Fase 1: Expansão do núcleo central de Fortaleza", "#475569"),
        (12500, 750, "Fase 1: Surge o núcleo denso do centro de Caucaia (Azul)", "#2563eb"),
        (11000, 750, "Fase 1: Consolidação dos núcleos urbanos históricos", "#475569"),
        (9500,  750, "Fase 1: Surge o núcleo denso de Maracanaú (Verde)", "#059669"),
        (8500,  750, "Fase 1: Três núcleos urbanos autônomos e isolados", "#475569"),
        (7500,  750, "Fase 1: Células urbanas preenchem as áreas centrais", "#475569"),
        (6800,  750, "Fase 1: Densificação ao longo dos eixos viários", "#475569"),
        (6000,  750, "Fase 1: Crescimento radial de Fortaleza em direção à periferia", "#475569"),
        (5400,  750, "Fase 1: Redução da distância livre entre Fortaleza e Caucaia", "#475569"),
        (4800,  750, "Fase 1: Franjas urbanas aproximam-se do limiar de contato", "#475569"),
    ]
    for d, ms, txt, cor in d_nucleos:
        cronograma.append(("nucleacao", d, ms, txt, cor))

    # 3. Fase 2: Primeira Conurbação — Fortaleza absorve Caucaia
    cronograma.append(("aproxima_caucaia", 4400, 800, "Aproximação: Células urbanas expandem-se ao longo da BR-222", "#2563eb"))
    cronograma.append(("aproxima_caucaia", 4200, 800, "Aproximação: Eixo litorâneo e rodoviário a menos de l = 1,8 km", "#2563eb"))
    # Ponto exato da conurbação de Caucaia: pausa de destaque de 3,0 segundos!
    cronograma.append(("conurba_caucaia", 4000, 3000, "[CONURBAÇÃO 1] Caucaia conecta-se a Fortaleza! (+360 mil hab)", "#2563eb"))
    cronograma.append(("pos_caucaia", 3800, 800, "Fusão consumada: Caucaia agora integra a mancha central", "#1e293b"))
    cronograma.append(("pos_caucaia", 3600, 800, "A mancha unificada avança rumo ao sul metropolitano", "#1e293b"))

    # 4. Fase 3: Segunda Conurbação — Absorção de Maracanaú
    cronograma.append(("aproxima_maracanau", 3400, 800, "Aproximação: Corredor industrial de Maracanaú expande-se ao sul", "#059669"))
    # Ponto exato da conurbação de Maracanaú: pausa de destaque de 3,0 segundos!
    cronograma.append(("conurba_maracanau", 3200, 3000, "[CONURBAÇÃO 2] Maracanaú integra-se à conurbação! (+230 mil hab)", "#059669"))
    cronograma.append(("pos_maracanau", 3000, 800, "Fusão consumada: Fortaleza, Caucaia e Maracanaú unidos", "#1e293b"))
    cronograma.append(("pos_maracanau", 2850, 800, "Pólo petroquímico e têxtil conurbados à capital", "#1e293b"))

    # 5. Fase 4: Expansão Periurbana Contínua e Fechamento (passos graduais até 1.450)
    d_periferia = [
        (2700, 750, "Fase 3: Expansão periurbana contígua e absorção de bordas", "#334155"),
        (2500, 750, "Fase 3: Adensamento contínuo em Messejana e corredor sudeste", "#334155"),
        (2300, 750, "Fase 3: Coalescência em direção aos limites de Eusébio", "#334155"),
        (2100, 750, "Fase 3: Franjas residenciais periféricas sendo incorporadas", "#334155"),
        (1950, 750, "Fase 3: Preenchimento de interstícios intrametropolitanos", "#334155"),
        (1800, 750, "Fase 3: Aproximação do limiar de calibração demográfica", "#334155"),
        (1680, 750, "Fase 3: Quase atingindo a mancha calibrada regional", "#334155"),
        (1560, 750, "Fase 3: Últimos agregados de menor densidade sendo capturados", "#334155"),
        (1490, 750, "Fase 3: Limiar imediatamente superior à calibração (D* = 1.490)", "#334155"),
    ]
    for d, ms, txt, cor in d_periferia:
        cronograma.append(("expansao", d, ms, txt, cor))

    # 6. Fase 5: Mancha Final Consolidada da Imagem (a) (D* = 1.450 hab/km²): pausa de 6,0 segundos!
    cronograma.append(("consolidado", 1450, 6000, "[CONFIGURAÇÃO DA IMAGEM a] Mancha Consolidada: 3,35M hab", "#047857"))

    def fig_to_rgb(fig):
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    frames_rgb = []
    durations_ms = []

    fig = plt.figure(figsize=(10.4, 5.5), dpi=120, facecolor="#ffffff")
    t_inicio = time.time()
    print(f"Total de etapas no cronograma: {len(cronograma)}")
    tempo_total_previsto = sum(item[2] for item in cronograma) / 1000.0
    print(f"Duração total prevista da animação: {tempo_total_previsto:.1f} segundos")

    for idx_etapa, (tipo_fase, D_val, dur_ms, txt_status, cor_destaque) in enumerate(cronograma):
        fig.clf()
        ax_map = fig.add_axes([0.05, 0.08, 0.52, 0.84])
        ax_info = fig.add_axes([0.62, 0.08, 0.34, 0.84])
        ax_info.axis("off")
        
        # Fundo do mapa: oceano azul claro límpido
        ax_map.set_facecolor("#e0f2fe")
        
        # -------------------------------------------------------------- Renderiza Mapa
        if tipo_fase == "intro_densidade":
            im = ax_map.imshow(np.where(sub_dens > 0, sub_dens, np.nan),
                               extent=extent, cmap="inferno", norm=LogNorm(100, 32000),
                               origin="upper", interpolation="bilinear", zorder=1)
            
            ax_map.text(-38.42, -3.66, "Oceano Atlântico", fontsize=7.5, fontstyle="italic",
                        color="#0284c7", zorder=4)
            
            for nome_p, (p_lo, p_la) in PONTOS_FOR.items():
                ax_map.plot(p_lo, p_la, "o", color="#38bdf8", ms=4.5, mec="#0f172a", mew=0.8, zorder=5)
                ax_map.annotate(nome_p, (p_lo, p_la), (4, 4), textcoords="offset points",
                                fontsize=6.5, fontweight="bold", color="#ffffff",
                                path_effects=[pe.withStroke(linewidth=2.2, foreground="#0f172a")],
                                zorder=6)
                
            ax_map.set_xlim(extent[0], extent[1])
            ax_map.set_ylim(extent[2], extent[3])
            ax_map.set_xticks([-38.7, -38.5])
            ax_map.set_xticklabels(["38°42'O", "38°30'O"], fontsize=7.2)
            ax_map.set_yticks([-4.0, -3.8, -3.65])
            ax_map.set_yticklabels(["4°00'S", "3°48'S", "3°39'S"], fontsize=7.2)
            
            ax_map.set_title("1. Densidade Demográfica Contínua (GPW v4, 2020)\nRegião Metropolitana de Fortaleza",
                             fontsize=8.5, fontweight="bold", color="#0f172a", pad=6)
            
            # Painel lateral informativo
            ax_info.add_patch(Rectangle((0, 0), 1, 1, transform=ax_info.transAxes,
                                        fc="#f8fafc", ec="#cbd5e1", lw=1.2))
            
            ax_info.text(0.5, 0.93, "RELEVO DEMOGRÁFICO DA RMF", transform=ax_info.transAxes,
                         ha="center", fontsize=8.2, fontweight="bold", color="#0369a1")
            
            ax_info.text(0.06, 0.81, "O raster demográfico global GPW v4 (30\")\nredistribui a população censitária numa grade\nregular de células de ~0,93 km.",
                         transform=ax_info.transAxes, fontsize=7.4, color="#334155", va="top")
            
            ax_info.text(0.06, 0.58, "Escala Térmica (inferno):\n• Escuro: baixas densidades rurais / mar\n• Amarelo/Branco: núcleos hiperdensos\n  (Centro, Aldeota, Caucaia, Maracanaú)",
                         transform=ax_info.transAxes, fontsize=7.4, color="#334155", va="top")
            
            ax_info.text(0.06, 0.35, "Princípio do Algoritmo CCA:\n1. Filtra células onde Densidade > D*\n2. Une células ativas com distância <= 1,8 km\n3. Revela as conurbações reais!",
                         transform=ax_info.transAxes, fontsize=7.4, color="#1e293b", va="top")
            
            ax_info.text(0.06, 0.12, "▶ A seguir: evolução passo a passo da conurbação...",
                         transform=ax_info.transAxes, fontsize=7.5, fontweight="bold", color="#ea580c")
            
        else:
            lab_full = V.rotulos(D_val, 1800)
            sub_lab = lab_full[r_min:r_max, c_min:c_max]
            
            # Fundo de relevo suave
            ax_map.imshow(np.where(sub_dens > 0, sub_dens, np.nan),
                          extent=extent, cmap="gray", norm=LogNorm(100, 32000),
                          alpha=0.18, origin="upper", zorder=1)
            
            ax_map.text(-38.42, -3.66, "Oceano Atlântico", fontsize=7.5, fontstyle="italic",
                        color="#0284c7", zorder=2)
            
            # Identificação das componentes-chave
            id_for = sub_lab[rf_peak, cf_peak]
            id_cauc = sub_lab[rc_peak, cc_peak]
            id_marac = sub_lab[rm_peak, cm_peak]
            
            cauc_unida = (id_cauc == id_for and id_for > 0)
            marac_unida = (id_marac == id_for and id_for > 0)
            
            is_final = (tipo_fase == "consolidado")
            
            # Camada RGB para as células ativas
            rgb_layer = np.zeros((sub_dens.shape[0], sub_dens.shape[1], 4))
            u_clusters = np.unique(sub_lab[sub_lab > 0])
            n_clusters = len(u_clusters)
            
            if is_final:
                for r_i in range(sub_lab.shape[0]):
                    for c_i in range(sub_lab.shape[1]):
                        cid = sub_lab[r_i, c_i]
                        if cid == id_fortaleza_calib:
                            rgb_layer[r_i, c_i] = [0.94, 0.45, 0.08, 0.95]  # Laranja âmbar da tese
                        elif cid > 0:
                            rgb_layer[r_i, c_i] = [0.55, 0.65, 0.75, 0.70]
            else:
                for r_i in range(sub_lab.shape[0]):
                    for c_i in range(sub_lab.shape[1]):
                        cid = sub_lab[r_i, c_i]
                        if cid == 0:
                            continue
                        if cid == id_for and id_for > 0:
                            rgb_layer[r_i, c_i] = [0.88, 0.11, 0.28, 0.95]  # Vermelho carmim de Fortaleza
                        elif cid == id_cauc and not cauc_unida:
                            rgb_layer[r_i, c_i] = [0.15, 0.39, 0.92, 0.95]  # Azul royal de Caucaia
                        elif cid == id_marac and not marac_unida:
                            rgb_layer[r_i, c_i] = [0.02, 0.59, 0.41, 0.95]  # Verde esmeralda de Maracanaú
                        else:
                            rgb_layer[r_i, c_i] = [0.85, 0.47, 0.02, 0.85]  # Outros fragmentos
                            
            ax_map.imshow(rgb_layer, extent=extent, origin="upper", interpolation="nearest", zorder=3)
            
            # Contorno da mancha consolidada no final
            if is_final:
                mask_for_final = (sub_lab == id_fortaleza_calib)
                ax_map.contour(mask_for_final, levels=[0.5], extent=extent,
                               colors=["#7c2d12"], linewidths=[1.4], origin="upper", zorder=4)
                
            # Marcadores urbanos com texto legível
            for nome_p, (p_lo, p_la) in PONTOS_FOR.items():
                ax_map.plot(p_lo, p_la, "o", color="#ffffff", ms=3.5, mec="#0f172a", mew=0.8, zorder=5)
                ax_map.annotate(nome_p, (p_lo, p_la), (4, 4), textcoords="offset points",
                                fontsize=6.2, fontweight="bold", color="#0f172a",
                                path_effects=[pe.withStroke(linewidth=2.0, foreground="#ffffff")],
                                zorder=6)
                
            ax_map.set_xlim(extent[0], extent[1])
            ax_map.set_ylim(extent[2], extent[3])
            ax_map.set_xticks([-38.7, -38.5])
            ax_map.set_xticklabels(["38°42'O", "38°30'O"], fontsize=7.2)
            ax_map.set_yticks([-4.0, -3.8, -3.65])
            ax_map.set_yticklabels(["4°00'S", "3°48'S", "3°39'S"], fontsize=7.2)
            
            d_fmt = f"{int(D_val):,}".replace(",", ".")
            titulo_mapa = ("[FIGURA a DA TESE] Mancha de Fortaleza Consolidada\n"
                           f"Configuração Calibrada: $D^* = 1.450$ hab/km² | $\\ell = 1,8$ km"
                           if is_final else
                           f"Conurbação Célula a Célula via CCA (RMF)\n"
                           f"Limiar: $D^* = {d_fmt}$ hab/km²  |  $\\ell = 1,8$ km")
            
            ax_map.set_title(titulo_mapa, fontsize=8.4, fontweight="bold",
                             color="#047857" if is_final else "#0f172a", pad=6)
            
            # Métricas da mancha de Fortaleza
            fort_mask = (sub_lab == (id_fortaleza_calib if is_final else id_for)) if (id_for > 0 or is_final) else np.zeros_like(sub_lab, dtype=bool)
            fort_pop = sub_pop[fort_mask].sum()
            fort_area = sub_area[fort_mask].sum()
            fort_n_px = fort_mask.sum()
            
            # ---------------------------------------------------------- Painel Lateral (HUD)
            bg_card = "#ecfdf5" if is_final else "#f8fafc"
            edge_card = "#059669" if is_final else "#cbd5e1"
            ax_info.add_patch(Rectangle((0, 0), 1, 1, transform=ax_info.transAxes,
                                        fc=bg_card, ec=edge_card, lw=1.2))
            
            hdr_txt = "[MANCHA CONSOLIDADA DA FIGURA a]" if is_final else "PROCESSO DE CONURBAÇÃO CCA"
            ax_info.text(0.5, 0.93, hdr_txt, transform=ax_info.transAxes,
                         ha="center", fontsize=8.0, fontweight="bold",
                         color="#047857" if is_final else "#0369a1")
            
            ax_info.text(0.06, 0.85, "Limiar de Densidade $D^*$:", transform=ax_info.transAxes,
                         fontsize=7.4, color="#475569")
            ax_info.text(0.94, 0.85, f"{d_fmt} hab/km²", transform=ax_info.transAxes,
                         fontsize=8.2, fontweight="bold", ha="right", color="#0f172a")
            
            pct_progresso = (16000.0 - D_val) / (16000.0 - 1450.0)
            pct_progresso = np.clip(pct_progresso, 0.0, 1.0)
            ax_info.add_patch(Rectangle((0.06, 0.80), 0.88, 0.025, transform=ax_info.transAxes,
                                        fc="#e2e8f0", ec="none"))
            ax_info.add_patch(Rectangle((0.06, 0.80), 0.88 * pct_progresso, 0.025, transform=ax_info.transAxes,
                                        fc="#059669" if is_final else "#ea580c", ec="none"))
            
            ax_info.text(0.06, 0.72, "População da Mancha:", transform=ax_info.transAxes,
                         fontsize=7.6, color="#475569")
            ax_info.text(0.94, 0.72, f"{fort_pop/1e6:.2f} M hab ({int(fort_pop):,})".replace(",", "."), transform=ax_info.transAxes,
                         fontsize=8.6, fontweight="bold", ha="right", color="#b91c1c")
            
            ax_info.text(0.06, 0.63, "Área Física Conurbada:", transform=ax_info.transAxes,
                         fontsize=7.6, color="#475569")
            ax_info.text(0.94, 0.63, f"{fort_area:.1f} km² ({fort_n_px} células)".replace(".", ","), transform=ax_info.transAxes,
                         fontsize=8.0, fontweight="bold", ha="right", color="#0f172a")
            
            ax_info.text(0.06, 0.54, "Fragmentos no Perímetro:", transform=ax_info.transAxes,
                         fontsize=7.6, color="#475569")
            clust_txt = "1 única mancha unificada!" if is_final else f"{n_clusters} fragmentos isolados"
            ax_info.text(0.94, 0.54, clust_txt, transform=ax_info.transAxes,
                         fontsize=8.0, fontweight="bold", ha="right",
                         color="#047857" if is_final else "#0f172a")
            
            # Caixa de Destaque Contextual
            box_destaque_fc = "#f0fdf4" if is_final else ("#eff6ff" if "Caucaia" in txt_status else ("#f0fdf4" if "Maracanaú" in txt_status else "#f8fafc"))
            box_destaque_ec = cor_destaque
            ax_info.add_patch(Rectangle((0.05, 0.36), 0.90, 0.14, transform=ax_info.transAxes,
                                        fc=box_destaque_fc, ec=box_destaque_ec, lw=1.2))
            
            ax_info.text(0.5, 0.43, txt_status, transform=ax_info.transAxes,
                         ha="center", va="center", fontsize=7.2, fontweight="bold",
                         color=cor_destaque)
            
            # Comparação administrativa IBGE
            ax_info.text(0.06, 0.30, "Comparação Administrativa (IBGE 2022):", transform=ax_info.transAxes,
                         fontsize=7.3, fontweight="bold", color="#1e293b")
            ax_info.text(0.06, 0.23, "• Município de Fortaleza: 2,43 M hab\n• Caucaia: 0,36 M | Maracanaú: 0,23 M\n• Eusébio: 0,07 M",
                         transform=ax_info.transAxes, fontsize=6.8, color="#334155", va="top")
            
            ax_info.text(0.06, 0.08, "▶ Conurbação real capturada pelo CCA:\n3,35 M hab (+38% maior que o município).",
                         transform=ax_info.transAxes, fontsize=6.8, fontweight="medium",
                         color="#047857" if is_final else "#475569", va="top")
            
        rgb = fig_to_rgb(fig)
        frames_rgb.append(rgb)
        durations_ms.append(dur_ms)

    plt.close(fig)
    print(f"Convertendo {len(frames_rgb)} quadros para Pillow Image...")
    pil_images = [Image.fromarray(f) for f in frames_rgb]

    print(f"Salvando GIF otimizado em: {out_path}...")
    pil_images[0].save(
        out_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=durations_ms,
        loop=0,
        optimize=False,
    )

    t_total = time.time() - t_inicio
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("Salvo com sucesso!")
    print(f"Arquivo: {out_path}")
    print(f"Tamanho: {file_size_mb:.2f} MB")
    print(f"Total de quadros únicos: {len(frames_rgb)}")
    print(f"Duração total da animação: {sum(durations_ms)/1000.0:.1f} segundos")
    return out_path


if __name__ == "__main__":
    gerar_gif_04_fortaleza_lento()
