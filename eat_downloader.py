from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


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


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, obj: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_download_plan(delta: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """
    Returns list of tuples: (kind, slug, filename, url)
    kind ∈ {"missing","changed"}
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
) -> Dict[str, Any]:
    """
    Downloads delta['missing'] and delta['changed'] into eat_dir.
    For 'changed', optionally archives existing file into eat_dir/archive/ with timestamp.
    Checkpoints after each success/failure to out_dir/<checkpoint_name> (atomic write).
    Resumes by skipping filenames already in checkpoint['downloaded'].
    """
    cfg = cfg or DownloadConfig()

    eat_dir = Path(eat_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / checkpoint_name
    archive_dir = (eat_dir / archive_subdir).resolve()

    # Resume state
    results = _load_json(checkpoint_path) or {"downloaded": [], "archived": [], "failed": []}
    if "downloaded" not in results:
        results["downloaded"] = []
    if "archived" not in results:
        results["archived"] = []
    if "failed" not in results:
        results["failed"] = []

    already_done = {Path(p).name for p in results.get("downloaded", [])}

    plan = build_download_plan(delta)
    if max_items is not None:
        plan = plan[:max_items]

    # tqdm total reflects planned actions (files)
    pbar = tqdm(plan, desc="Downloading EAT PDFs", unit="file", total=len(plan))

    for kind, slug, filename, url in pbar:
        if filename in already_done:
            continue

        dest = eat_dir / filename

        try:
            if kind == "changed" and archive_changed and dest.exists():
                archived = _archive_existing(dest, archive_dir, slug)
                if archived:
                    results["archived"].append(str(archived))
                    _write_json_atomic(checkpoint_path, results)

            _download_file(url, dest, cfg)
            results["downloaded"].append(str(dest))
            _write_json_atomic(checkpoint_path, results)

            already_done.add(filename)

        except Exception as e:
            results["failed"].append(
                {
                    "kind": kind,
                    "slug": slug,
                    "filename": filename,
                    "url": url,
                    "error": str(e),
                }
            )
            _write_json_atomic(checkpoint_path, results)

    return results
