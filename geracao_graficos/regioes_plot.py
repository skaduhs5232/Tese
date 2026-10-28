"""
Shared map extents and English labels for the 6 demographic regions used by the figure scripts.
The hydrology runs on 2 contiguous blocks; these boxes only frame the regional map panels.
"""

REGIOES_PLOT = [
    {"id": "americas_norte_central_caribe", "label": "North & Central America", "box": (-170.0, -2.0, -50.0, 75.0)},
    {"id": "america_do_sul", "label": "South America", "box": (-92.0, -56.0, -32.0, 13.0)},
    {"id": "europa_mediterraneo", "label": "Europe (to the Urals)", "box": (-25.0, 34.0, 60.0, 72.0)},
    {"id": "africa_madagascar", "label": "Africa & Madagascar", "box": (-26.0, -36.0, 58.0, 38.0)},
    {"id": "asia_continental_arquipelagos", "label": "Asia", "box": (25.0, -24.0, 180.0, 78.0)},
    {"id": "oceania_australia", "label": "Australia & Oceania", "box": (110.0, -48.0, 180.0, 0.0)},
]

LABEL_REGIAO = {r["id"]: r["label"] for r in REGIOES_PLOT}
