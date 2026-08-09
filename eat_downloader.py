from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from json_utils import load_json, write_json_atomic


try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


@dataclass
class DownloadConfig:
    timeout: int = 60
    min_delay: float = 0.2
    max_retries: int = 4
    backoff_base: float = 0.8
    chunk_size: int = 1024 * 1024  # 1MB


def _safe_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _download_file(url: str, dest: Path, cfg: DownloadConfig) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Optional[Exception] = None
    for attempt in range(cfg.max_retries + 1):
        try:
            time.sleep(cfg.min_delay)
            with requests.get(url, stream=True, timeout=cfg.timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=cfg.chunk_size):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            return
        except Exception as e:
            last_exc = e
            if attempt >= cfg.max_retries:
                raise
            time.sleep(cfg.backoff_base * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


def _archive_existing(local_path: Path, archive_dir: Path, slug: str) -> Optional[Path]:
    if not local_path.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = _safe_ts()
    archived_name = f"{local_path.stem}__{slug}__archived_{ts}{local_path.suffix}"
    archived_path = archive_dir / archived_name
    shutil.move(str(local_path), str(archived_path))
    return archived_path


def build_download_plan(delta: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """
    Returns list of tuples: (kind, slug, filename, url)
    kind in {"missing", "changed"}
    """
    plan: List[Tuple[str, str, str, str]] = []

    for item in (delta.get("missing") or []):
        fn = item.get("filename")
        url = item.get("pdf_url")
        slug = item.get("slug")
        if fn and url and slug:
            plan.append(("missing", slug, fn, url))

    for item in (delta.get("changed") or []):
        fn = item.get("filename")
        url = item.get("pdf_url")
        slug = item.get("slug")
        if fn and url and slug:
            plan.append(("changed", slug, fn, url))

    return plan


def download_missing_and_changed(
    *,
    delta: Dict[str, Any],
    eat_dir: Path,
    out_dir: Path,
    archive_changed: bool = True,
    checkpoint_name: str = "download_checkpoint.json",
    archive_subdir: str = "archive",
    cfg: Optional[DownloadConfig] = None,
    max_items: Optional[int] = None,
    max_workers: int = 8,
    checkpoint_every: int = 20,
) -> Dict[str, Any]:
    """
    Downloads delta['missing'] and delta['changed'] into eat_dir, in parallel
    (max_workers concurrent downloads; each file still respects cfg.min_delay).
    For 'changed', optionally archives existing file into eat_dir/archive/ with timestamp.
    Checkpoints to out_dir/<checkpoint_name> (atomic write) every checkpoint_every
    completions, plus once more at the end, so at most checkpoint_every already-
    downloaded files would be redundantly retried if interrupted mid-run.
    Resumes by skipping filenames already in checkpoint['downloaded'] or ['failed'].
    """
    cfg = cfg or DownloadConfig()

    eat_dir = Path(eat_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / checkpoint_name
    archive_dir = (eat_dir / archive_subdir).resolve()

    # Resume state
    results = load_json(checkpoint_path, default={}) or {"downloaded": [], "archived": [], "failed": []}
    for key in ("downloaded", "archived", "failed"):
        if key not in results:
            results[key] = []

    already_done = {Path(p).name for p in results.get("downloaded", [])}
    already_failed = {item["filename"] for item in results.get("failed", [])}

    plan = build_download_plan(delta)
    if max_items is not None:
        plan = plan[:max_items]

    pending = [
        (kind, slug, filename, url)
        for kind, slug, filename, url in plan
        if filename not in already_done and filename not in already_failed
    ]
    already_settled = len(plan) - len(pending)

    lock = threading.Lock()
    pending_writes = 0

    def _checkpoint(force: bool = False) -> None:
        nonlocal pending_writes
        with lock:
            pending_writes += 1
            if force or pending_writes >= checkpoint_every:
                write_json_atomic(checkpoint_path, results)
                pending_writes = 0

    def _process_one(kind: str, slug: str, filename: str, url: str) -> None:
        dest = eat_dir / filename

        if kind == "changed" and archive_changed and dest.exists():
            archived = _archive_existing(dest, archive_dir, slug)
            if archived:
                with lock:
                    results["archived"].append(str(archived))

        _download_file(url, dest, cfg)

        with lock:
            results["downloaded"].append(str(dest))

    # tqdm total reflects planned actions; already-settled files count as pre-done
    pbar = tqdm(total=len(plan), initial=already_settled, desc="Downloading PDFs", unit="file")

    try:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futures = {
                ex.submit(_process_one, kind, slug, filename, url): (kind, slug, filename, url)
                for kind, slug, filename, url in pending
            }

            for fut in as_completed(futures):
                kind, slug, filename, url = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    with lock:
                        results["failed"].append(
                            {
                                "kind": kind,
                                "slug": slug,
                                "filename": filename,
                                "url": url,
                                "error": str(e),
                            }
                        )
                pbar.update(1)
                _checkpoint()
    finally:
        _checkpoint(force=True)
        pbar.close()

    return results
