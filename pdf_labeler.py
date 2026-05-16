from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LABEL_KEYS = [
    "UNFAIR_DISMISSAL",
    "CONSTRUCTIVE_DISMISSAL",
    "WRONGFUL_DISMISSAL",
    "WHISTLEBLOWING_PIDA",
    "DISCRIMINATION",
    "REDUNDANCY",
    "TUPE",
]

RULES: Dict[str, re.Pattern[str]] = {
    "UNFAIR_DISMISSAL": re.compile(
        r"\bunfair dismissal\b|\bera\s*1996\b|\bsection\s*98\b|"
        r"\bs\.?\s*98\b|\bs\.?\s*98\s*\(\s*4\s*\)\b",
        re.I,
    ),
    "CONSTRUCTIVE_DISMISSAL": re.compile(r"\bconstructive dismissal\b", re.I),
    "WRONGFUL_DISMISSAL": re.compile(r"\bwrongful dismissal\b", re.I),
    "WHISTLEBLOWING_PIDA": re.compile(
        r"\bwhistleblow|\bprotected disclosure\b|\bpida\b|"
        r"\bpublic interest disclosure\b",
        re.I,
    ),
    "DISCRIMINATION": re.compile(
        r"\bequality act\b|\beqa\s*2010\b|\bdiscrimination\b|"
        r"\bharass\w*\b|\bvictimisa\w*\b",
        re.I,
    ),
    "REDUNDANCY": re.compile(r"\bredundan\w*\b", re.I),
    "TUPE": re.compile(r"\btupe\b|\btransfer of undertakings\b", re.I),
}

DEFAULT_PDF_DIRS = {
    "EAT": Path("/media/hello/Vault/Tribunals/EAT_Appeals"),
    "ET": Path("/media/hello/Vault/Tribunals/ET_Cases"),
}

REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class PDFLabelConfig:
    mode: str = "ET"
    pdf_root: Optional[Path] = None
    out_dir: Optional[Path] = None
    first_n_pages: int = 2
    max_workers: Optional[int] = None
    chunk_print_every: int = 2000

    def __post_init__(self) -> None:
        self.mode = self.mode.upper()
        if self.mode not in DEFAULT_PDF_DIRS:
            raise ValueError("mode must be 'EAT' or 'ET'")

        if self.pdf_root is None:
            self.pdf_root = DEFAULT_PDF_DIRS[self.mode]
        self.pdf_root = Path(self.pdf_root).expanduser().resolve()

        if self.out_dir is None:
            self.out_dir = REPO_ROOT / "indexes" / self.mode
        self.out_dir = Path(self.out_dir).expanduser().resolve()

        if self.max_workers is None:
            default_workers = min(16, max(1, (os.cpu_count() or 1) // 2))
            self.max_workers = int(os.environ.get("PDF_WORKERS", default_workers))

    @property
    def cache_path(self) -> Path:
        return self.out_dir / f"cache__{self.mode}.json"

    @property
    def labels_path(self) -> Path:
        return self.out_dir / f"labels__{self.mode}.jsonl"

    @property
    def failures_path(self) -> Path:
        return self.out_dir / f"failures__{self.mode}.jsonl"

    @property
    def stats_path(self) -> Path:
        return self.out_dir / f"stats__{self.mode}.json"


def load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def iter_pdfs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.pdf"))


def file_sig(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime)


def classify_text(text: str) -> Dict[str, bool]:
    if not text:
        return {key: False for key in LABEL_KEYS}
    return {key: bool(RULES[key].search(text)) for key in LABEL_KEYS}


def extract_first_pages_text(pdf_path: str, first_n_pages: int) -> str:
    import fitz  # type: ignore

    doc = fitz.open(pdf_path)
    try:
        page_count = min(first_n_pages, doc.page_count)
        parts = []
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            parts.append(page.get_text("text") or "")
        return "\n".join(parts)
    finally:
        doc.close()


def worker(pdf_path: str, first_n_pages: int) -> Dict[str, Any]:
    started_at = time.time()
    try:
        text = extract_first_pages_text(pdf_path, first_n_pages)
        labels = classify_text(text)
        return {
            "path": pdf_path,
            "labels": labels,
            "text_ok": True,
            "text_chars": len(text),
            "word_count": len(text.split()),
            "elapsed_s": round(time.time() - started_at, 4),
        }
    except Exception as exc:
        return {
            "path": pdf_path,
            "labels": {key: False for key in LABEL_KEYS},
            "text_ok": False,
            "word_count": 0,
            "error": repr(exc),
            "elapsed_s": round(time.time() - started_at, 4),
        }


def label_pdfs(config: PDFLabelConfig) -> Dict[str, Any]:
    if not config.pdf_root.exists():
        raise FileNotFoundError(f"[{config.mode}] PDF root does not exist: {config.pdf_root}")
    if not config.pdf_root.is_dir():
        raise NotADirectoryError(f"[{config.mode}] PDF root is not a directory: {config.pdf_root}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(config.cache_path)
    pdfs = iter_pdfs(config.pdf_root)
    total = len(pdfs)

    print(f"[{config.mode}] PDF_ROOT={config.pdf_root}")
    print(f"[{config.mode}] PDFs found: {total:,}")
    print(
        f"[{config.mode}] MAX_WORKERS={config.max_workers} "
        f"FIRST_N_PAGES={config.first_n_pages}"
    )
    print(f"[{config.mode}] OUT_DIR={config.out_dir}")

    tasks: List[Path] = []
    reused = 0
    for pdf in pdfs:
        signature = file_sig(pdf)
        key = str(pdf)
        previous = cache.get(key)
        if previous and previous.get("sig") == list(signature):
            reused += 1
        else:
            tasks.append(pdf)

    print(f"[{config.mode}] Cache reused: {reused:,}")
    print(f"[{config.mode}] To process:   {len(tasks):,}")

    results: Dict[str, Any] = {}
    for pdf in pdfs:
        key = str(pdf)
        previous = cache.get(key)
        if previous and "result" in previous:
            results[key] = previous["result"]

    processed = 0
    started_at = time.time()

    if tasks:
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(worker, str(pdf), config.first_n_pages): str(pdf)
                for pdf in tasks
            }
            for future in as_completed(futures):
                path = futures[future]
                result = future.result()
                results[path] = result
                cache[path] = {"sig": list(file_sig(Path(path))), "result": result}
                processed += 1

                if processed % config.chunk_print_every == 0:
                    elapsed = time.time() - started_at
                    print(
                        f"[{config.mode}] processed {processed:,}/"
                        f"{len(tasks):,} new in {elapsed:.1f}s"
                    )

    ok_count = 0
    hit_counts = {key: 0 for key in LABEL_KEYS}
    text_fail = 0

    with (
        config.labels_path.open("w", encoding="utf-8") as labels_file,
        config.failures_path.open("w", encoding="utf-8") as failures_file,
    ):
        for pdf in pdfs:
            key = str(pdf)
            result = results.get(key)
            if not result:
                continue

            if not result.get("text_ok"):
                text_fail += 1
                failures_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                ok_count += 1

            labels = result.get("labels") or {}
            for label in LABEL_KEYS:
                if labels.get(label):
                    hit_counts[label] += 1

            labels_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    save_cache(config.cache_path, cache)

    stats = {
        "mode": config.mode,
        "pdf_root": str(config.pdf_root),
        "total_pdfs": total,
        "text_ok": ok_count,
        "text_fail": text_fail,
        "hits": hit_counts,
        "first_n_pages": config.first_n_pages,
        "max_workers": config.max_workers,
        "labels_path": str(config.labels_path),
        "fails_path": str(config.failures_path),
        "cache_path": str(config.cache_path),
        "elapsed_s": round(time.time() - started_at, 2),
    }
    config.stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[{config.mode}] DONE.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def run_pdf_labeling(**kwargs: Any) -> Dict[str, Any]:
    return label_pdfs(PDFLabelConfig(**kwargs))
