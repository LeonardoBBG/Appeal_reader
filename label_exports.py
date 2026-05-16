from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import polars as pl

from pdf_labeler import LABEL_KEYS, REPO_ROOT


def _flatten_labels(df: pl.DataFrame) -> pl.DataFrame:
    label_fields = df.select("labels").schema["labels"].fields
    label_names = [field.name for field in label_fields]
    return df.with_columns(
        [pl.col("labels").struct.field(name).alias(name) for name in label_names]
    ).drop("labels")


def _detect_path_col(df: pl.DataFrame, path_in: Path) -> str:
    path_candidates = [
        "path",
        "pdf_path",
        "file_path",
        "filepath",
        "filename",
        "source_path",
    ]
    path_col = next((col for col in path_candidates if col in df.columns), None)
    if path_col is None:
        raise ValueError(f"No path-like column found in {path_in}. Columns: {df.columns}")
    return path_col


def _normalize_label_bools(df: pl.DataFrame) -> pl.DataFrame:
    missing = [key for key in LABEL_KEYS if key not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    return df.with_columns(
        [
            pl.col(key)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.to_lowercase()
            .is_in(["true", "1", "yes", "y"])
            .alias(key)
            for key in LABEL_KEYS
        ]
    )


def build_wide_csv(df: pl.DataFrame, path_col: str, out_csv: Path) -> Dict[str, int]:
    lists = {
        key: (
            df.filter(pl.col(key) == True)
            .select(pl.col(path_col).cast(pl.Utf8))
            .to_series()
            .to_list()
        )
        for key in LABEL_KEYS
    }

    max_len = max((len(paths) for paths in lists.values()), default=0)
    wide_df = pl.DataFrame(
        {key: paths + [None] * (max_len - len(paths)) for key, paths in lists.items()}
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_df.write_csv(out_csv)
    return {key: len(paths) for key, paths in lists.items()}


def write_counts_csv(counts: Dict[str, int], court: str, out_csv: Path) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "court": [court] * len(counts),
            "label": list(counts.keys()),
            "count": list(counts.values()),
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_csv)
    return df


def build_long_df(df: pl.DataFrame, path_col: str, court: str) -> pl.DataFrame:
    parts = []
    for key in LABEL_KEYS:
        parts.append(
            df.filter(pl.col(key) == True).select(
                [
                    pl.lit(court).alias("court"),
                    pl.lit(key).alias("label"),
                    pl.col(path_col).cast(pl.Utf8).alias("path"),
                ]
            )
        )
    return pl.concat(parts, how="vertical")


def export_label_csvs(
    *,
    index_root: Optional[Path] = None,
    courts: Iterable[str] = ("EAT", "ET"),
) -> Dict[str, Path]:
    index_root = Path(index_root or (REPO_ROOT / "indexes")).expanduser().resolve()
    long_parts: List[pl.DataFrame] = []
    count_parts: List[pl.DataFrame] = []

    for court in courts:
        court = court.upper()
        path_in = index_root / court / f"labels__{court}.jsonl"
        out_wide = index_root / court / f"paths_by_label__WIDE__{court}.csv"
        out_counts = index_root / court / f"paths_by_label__COUNTS__{court}.csv"

        print(f"\nProcessing {court}...")
        df = pl.read_ndjson(path_in)
        df = _flatten_labels(df)
        path_col = _detect_path_col(df, path_in)
        df = _normalize_label_bools(df)

        counts = build_wide_csv(df, path_col, out_wide)
        counts_df = write_counts_csv(counts, court, out_counts)

        print(f"Saved wide:   {out_wide}")
        print(f"Saved counts: {out_counts}")

        long_parts.append(build_long_df(df, path_col, court))
        count_parts.append(counts_df)

    out_long_master = index_root / "paths_by_label__LONG__EAT_ET.csv"
    out_counts_master = index_root / "paths_by_label__COUNTS__EAT_ET.csv"

    master_long = (
        pl.concat(long_parts, how="vertical")
        if long_parts
        else pl.DataFrame({"court": [], "label": [], "path": []})
    )
    master_counts = (
        pl.concat(count_parts, how="vertical")
        if count_parts
        else pl.DataFrame({"court": [], "label": [], "count": []})
    )

    master_long.write_csv(out_long_master)
    master_counts.write_csv(out_counts_master)

    print(f"\nSaved master long:   {out_long_master}")
    print(f"Saved master counts: {out_counts_master}")
    print(f"Rows (long): {master_long.height}")

    return {
        "master_long": out_long_master,
        "master_counts": out_counts_master,
    }
