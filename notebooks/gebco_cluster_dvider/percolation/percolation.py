import sys
from pathlib import Path

# Garante que o pacote raiz `calculos/` possa ser importado a partir de qualquer diretório de trabalho
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from calculos.percolacao_ilhas.percolacao import Percolation, extrair_recorte_cluster

__all__ = ["Percolation", "extrair_recorte_cluster"]
