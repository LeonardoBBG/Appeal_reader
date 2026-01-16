# -----------------------------
# Local PDF index + delta utils
# -----------------------------
from pathlib import Path
import hashlib
from typing import Dict, Any, Optional
import json
from pathlib import Path


def scan_local_pdfs(
    root_dir: Path,
    recursive: bool = True,
    compute_sha256: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Index local PDFs under root_dir.

    Returns dict keyed by filename (not slug), with:
      path, filename, size, mtime, sha256 (optional)
    """
    root_dir = Path(root_dir).expanduser().resolve()
    pattern = "**/*.pdf" if recursive else "*.pdf"

    out: Dict[str, Dict[str, Any]] = {}
    for p in root_dir.glob(pattern):
        if not p.is_file():
            continue

        try:
            stat = p.stat()
        except FileNotFoundError:
            continue

        rec: Dict[str, Any] = {
            "path": str(p.resolve()),
            "filename": p.name,
            "size": int(stat.st_size),
            "mtime": float(stat.st_mtime),
            "sha256": None,
        }

        if compute_sha256:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            rec["sha256"] = h.hexdigest()

        # key by filename (practical for matching pdf_filename from remote_index)
        out[p.name] = rec

    return out


def write_json(path: Path, obj):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def compute_delta(
    remote_index: Dict[str, Dict[str, Any]],
    local_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute missing/changed/orphaned comparing:
      remote_index keyed by slug (remote_index[slug]["pdf_filename"])
      local_index keyed by filename

    "changed" uses remote pdf_content_length if present (from HEAD),
    otherwise it cannot reliably detect changes without sha256.
    """
    # Map remote filename -> remote record + slug
    remote_by_filename: Dict[str, Dict[str, Any]] = {}
    for slug, r in (remote_index or {}).items():
        fn = r.get("pdf_filename")
        if fn:
            remote_by_filename[fn] = {"slug": slug, **r}

    remote_files = set(remote_by_filename.keys())
    local_files = set((local_index or {}).keys())

    missing_files = sorted(remote_files - local_files)
    orphaned_files = sorted(local_files - remote_files)

    changed = []
    for fn in sorted(remote_files & local_files):
        r = remote_by_filename[fn]
        local_sz = local_index[fn].get("size")
        remote_sz = r.get("pdf_content_length")

        # only flag if we can compare numbers
        if remote_sz is not None and local_sz is not None:
            try:
                if int(remote_sz) != int(local_sz):
                    changed.append(
                        {
                            "filename": fn,
                            "slug": r.get("slug"),
                            "remote_size": int(remote_sz),
                            "local_size": int(local_sz),
                            "pdf_url": r.get("pdf_url"),
                        }
                    )
            except Exception:
                pass

    delta = {
        "counts": {
            "remote": len(remote_files),
            "local": len(local_files),
            "missing": len(missing_files),
            "changed": len(changed),
            "orphaned": len(orphaned_files),
        },
        "missing": [
            {
                "filename": fn,
                "slug": remote_by_filename[fn].get("slug"),
                "pdf_url": remote_by_filename[fn].get("pdf_url"),
            }
            for fn in missing_files
        ],
        "changed": changed,
        "orphaned": [{"filename": fn, "path": local_index[fn].get("path")} for fn in orphaned_files],
    }
    return delta
