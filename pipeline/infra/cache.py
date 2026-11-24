"""Persistência padronizada de caches CSV e NPZ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ler_csv_cache(path: str | Path) -> pd.DataFrame | None:
    """Lê um CSV de cache se existir e não estiver vazio."""
    p = Path(path)
    return pd.read_csv(p) if p.is_file() and p.stat().st_size > 10 else None


def salvar_csv_cache(df: pd.DataFrame, path: str | Path) -> None:
    """Salva DataFrame em CSV criando diretórios pai automaticamente."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def ler_npz_cache(path: str | Path | None) -> dict[str, np.ndarray] | None:
    """Lê arrays de um arquivo NPZ se existir."""
    if path is None or not Path(path).is_file():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def salvar_npz_cache(path: str | Path | None, **arrays: Any) -> None:
    """Salva arrays comprimidos em NPZ se um caminho for informado."""
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
