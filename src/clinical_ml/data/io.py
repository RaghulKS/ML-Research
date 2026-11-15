"""CSV and Parquet readers with encoding fallbacks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: Path | str, chunksize: int | None = 200_000) -> pd.DataFrame:
    """Read a table. CSV columns stay strings so ICD codes are not recast to floats."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        last_error: Exception | None = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                if chunksize:
                    chunks = list(pd.read_csv(path, dtype=str, encoding=encoding, chunksize=chunksize))
                    if not chunks:
                        return pd.DataFrame()
                    return pd.concat(chunks, ignore_index=True)
                return pd.read_csv(path, dtype=str, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error:
            raise last_error
        raise OSError(f"Could not read {path}")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table type '{suffix}'. Use CSV or Parquet.")


def discover_file(directory: Path, stems: list[str]) -> Path | None:
    lookup = {}
    if not directory.exists():
        return None
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".pq"}:
            lookup[path.stem.lower()] = path
    for stem in stems:
        if stem.lower() in lookup:
            return lookup[stem.lower()]
    return None


def pick_column(frame: pd.DataFrame | None, names: list[str], required: bool = True) -> str | None:
    if frame is None:
        if required:
            raise KeyError(f"Missing table while looking for {names}")
        return None
    lookup = {column.lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise KeyError(f"None of {names} found in columns {list(frame.columns)}")
    return None
