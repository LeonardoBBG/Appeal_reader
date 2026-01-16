from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote
import tqdm 

import requests


# -----------------------------
# HTTP client (robust-ish)
# -----------------------------
class HttpClient:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff_base: float = 0.8,
        min_delay: float = 0.15,
        user_agent: str = "Mozilla/5.0 (compatible; EATIndexer/2.0; +local)",
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.min_delay = min_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _sleep_polite(self) -> None:
        if self.min_delay > 0:
            time.sleep(self.min_delay)

    def get_json(self, url: str) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._sleep_polite()
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"GET {url} -> {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.backoff_base * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    def head(self, url: str) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._sleep_polite()
                resp = self.session.head(url, timeout=self.timeout, allow_redirects=True)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HEAD {url} -> {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    # Don't hard-fail the whole run on HEAD issues
                    return _dummy_head_response(url, error=str(e))
                time.sleep(self.backoff_base * (2 ** attempt))
        return _dummy_head_response(url, error=str(last_exc))


def _dummy_head_response(url: str, error: str) -> requests.Response:
    resp = requests.Response()
    resp.status_code = 0
    resp.url = url
    resp._content = b""
    resp.headers["x-head-error"] = error
    return resp


def _safe_int(x: Optional[str]) -> Optional[int]:
    if not x:
        return None
    try:
        return int(x)
    except Exception:
        return None

def iter_search_results_with_tqdm(
    client: HttpClient,
    *,
    doc_type: str,
    count: int = 200,
    max_items: int | None = None,
    order: str = "-public_timestamp",
):
    """
    Paginated GOV.UK Search API iterator with tqdm.
    doc_type must be explicitly provided, e.g.:
      - employment_appeal_tribunal_decision (EAT)
      - employment_tribunal_decision (ET)
    """
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **kwargs):
            return x

    start = 0
    yielded = 0
    pbar = None

    while True:
        url = (
            f"{GOVUK}/api/search.json"
            f"?filter_document_type={doc_type}"
            f"&order={order}"
            f"&count={count}"
            f"&start={start}"
        )
        payload = client.get_json(url)

        results = payload.get("results") or []
        if not results:
            break

        # Initialise tqdm lazily once total is known
        if pbar is None:
            total = payload.get("total")
            pbar = tqdm(
                total=total,
                desc=f"Indexing {doc_type}",
                unit="doc",
            )

        for r in results:
            yield r
            yielded += 1
            pbar.update(1)

            if max_items is not None and yielded >= max_items:
                pbar.close()
                return

        start += len(results)
        total = payload.get("total")
        if isinstance(total, int) and start >= total:
            break

    if pbar:
        pbar.close()

# -----------------------------
# Data model
# -----------------------------
@dataclass
class RemoteDecision:
    slug: str
    decision_page_url: str
    title: str
    decision_date: Optional[str] = None
    published_date: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_filename: Optional[str] = None

    # HEAD-derived (optional)
    pdf_etag: Optional[str] = None
    pdf_last_modified: Optional[str] = None
    pdf_content_length: Optional[int] = None
    head_error: Optional[str] = None


# -----------------------------
# GOV.UK APIs
# -----------------------------
GOVUK = "https://www.gov.uk"
SEARCH_API = f"{GOVUK}/api/search.json"
DOC_TYPE = "employment_appeal_tribunal_decision"


def _slug_from_decision_page(url: str) -> str:
    # https://www.gov.uk/employment-appeal-tribunal-decisions/<slug>
    path = urlparse(url).path.rstrip("/")
    return path.split("/employment-appeal-tribunal-decisions/")[-1].strip("/")


def _pick_pdf_from_content_api(content: Dict[str, Any]) -> Optional[str]:
    """
    Content API payload can include attachments in different structures across formats.
    We keep it defensive and just hunt for the first .pdf URL.
    """
    def walk(o: Any) -> Iterable[str]:
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    yield from walk(v)
                elif isinstance(v, str):
                    yield v
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)

    for s in walk(content):
        if isinstance(s, str) and ".pdf" in s.lower():
            # Most attachment URLs are already absolute; if not, join.
            if s.startswith("http://") or s.startswith("https://"):
                return s
            return urljoin(GOVUK, s)
    return None



def build_remote_index_v2(
    client: HttpClient,
    *,
    do_head: bool = True,
    head_only_assets: bool = True,
    max_items: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Builds slug-keyed index via Search API + Content API.
    """
    out: Dict[str, Dict[str, Any]] = {}

    for i, r in enumerate(iter_search_results_with_tqdm(client, max_items=max_items), start=1):
        link = r.get("link")  # usually "/employment-appeal-tribunal-decisions/<slug>"
        if not link:
            continue
        decision_url = urljoin(GOVUK, link)
        slug = _slug_from_decision_page(decision_url)

        # Pull richer metadata (attachments, exact fields) from Content API
        content_url = urljoin(GOVUK, "/api/content" + link)
        content = client.get_json(content_url)

        title = (content.get("title") or r.get("title") or "").strip()

        # Dates: keep both if present; formats can vary, so we store raw strings
        published_date = content.get("public_updated_at") or r.get("public_timestamp")
        decision_date = None

        pdf_url = _pick_pdf_from_content_api(content)
        pdf_filename = Path(urlparse(pdf_url).path).name if pdf_url else None

        rec = RemoteDecision(
            slug=slug,
            decision_page_url=decision_url,
            title=title,
            decision_date=decision_date,
            published_date=published_date,
            pdf_url=pdf_url,
            pdf_filename=pdf_filename,
        )

        if do_head and pdf_url:
            if (not head_only_assets) or ("assets.publishing.service.gov.uk" in pdf_url):
                h = client.head(pdf_url)
                rec.pdf_etag = h.headers.get("ETag")
                rec.pdf_last_modified = h.headers.get("Last-Modified")
                rec.pdf_content_length = _safe_int(h.headers.get("Content-Length"))
                rec.head_error = h.headers.get("x-head-error")

        out[slug] = dataclasses.asdict(rec)

        if i % 100 == 0:
            print(f"[remote] processed {i} decisions...")

    return out


# -----------------------------
# JSON helpers
# -----------------------------
def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
