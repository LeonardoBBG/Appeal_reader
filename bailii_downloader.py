from __future__ import annotations

import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from json_utils import load_json, write_json_atomic


try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):  # type: ignore
        return x


BAILII_BASE_URL = "https://www.bailii.org"

BAILII_PATHS = {
    "EAT": "/uk/cases/UKEAT/",
    "ET": "/uk/cases/UKET/",
}

DEFAULT_ROOTS = {
    "EAT": Path("/media/hello/Vault/Tribunals/EAT_Bailii"),
    "ET": Path("/media/hello/Vault/Tribunals/ET_Bailii"),
}

RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class BailiiConfig:
    mode: str = "ET"
    start_year: int = 2022
    end_year: int = 2026
    root_dir: Optional[Path] = None
    max_threads: int = 20
    max_cases_per_year: Optional[int] = None
    max_retries: int = 4
    timeout: int = 20
    sleep_min: float = 0.05
    sleep_max: float = 0.20
    checkpoint_every: int = 50
    error_rate_threshold: float = 0.20
    throttle_multiplier: float = 1.50
    max_sleep_cap: float = 1.50
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    )

    def __post_init__(self) -> None:
        self.mode = self.mode.upper()
        if self.mode not in BAILII_PATHS:
            raise ValueError("mode must be 'EAT' or 'ET'")
        if self.start_year > self.end_year:
            raise ValueError("start_year must be less than or equal to end_year")
        if self.root_dir is None:
            self.root_dir = DEFAULT_ROOTS[self.mode]
        self.root_dir = Path(self.root_dir).expanduser().resolve()
        if self.max_cases_per_year is not None and self.max_cases_per_year < 1:
            raise ValueError("max_cases_per_year must be at least 1 when set")

    @property
    def base_path(self) -> str:
        return BAILII_PATHS[self.mode]

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / f"{self.mode.lower()}_bailii_manifest.json"

    @property
    def failed_urls_path(self) -> Path:
        return self.root_dir / f"{self.mode.lower()}_bailii_failed_urls.json"


def summarize_bailii_root(mode: str, root_dir: Optional[Path] = None) -> Dict[str, Any]:
    mode = mode.upper()
    if mode not in BAILII_PATHS:
        raise ValueError("mode must be 'EAT' or 'ET'")

    root = Path(root_dir or DEFAULT_ROOTS[mode]).expanduser().resolve()
    manifest_path = root / f"{mode.lower()}_bailii_manifest.json"
    failed_urls_path = root / f"{mode.lower()}_bailii_failed_urls.json"

    years = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else []
    html_count = sum(1 for _ in root.glob("*/*.html")) if root.exists() else 0
    text_count = sum(1 for _ in root.glob("*/*.txt")) if root.exists() else 0
    manifest = load_json(manifest_path, default={})
    failed_urls = load_json(failed_urls_path, default=[])

    return {
        "mode": mode,
        "root_dir": str(root),
        "exists": root.exists(),
        "year_start": years[0] if years else None,
        "year_end": years[-1] if years else None,
        "year_dirs": len(years),
        "html_files": html_count,
        "text_files": text_count,
        "manifest_path": str(manifest_path),
        "manifest_records": len(manifest) if isinstance(manifest, dict) else 0,
        "failed_urls_path": str(failed_urls_path),
        "failed_urls": len(failed_urls) if isinstance(failed_urls, list) else 0,
    }


class BailiiDownloader:
    def __init__(self, config: BailiiConfig) -> None:
        self.config = config
        self.thread_local = threading.local()
        self.adaptive_sleep_lock = threading.Lock()
        self.manifest_lock = threading.Lock()
        self.failures_lock = threading.Lock()
        self.adaptive_sleep_min = config.sleep_min
        self.adaptive_sleep_max = config.sleep_max

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        cfg.root_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{cfg.mode}] Root directory: {cfg.root_dir}")
        print(f"[{cfg.mode}] Manifest path : {cfg.manifest_path}")
        print(f"[{cfg.mode}] Failed path   : {cfg.failed_urls_path}")
        print(f"[{cfg.mode}] Years         : {cfg.start_year} -> {cfg.end_year}")
        print(f"[{cfg.mode}] Max threads   : {cfg.max_threads}")
        if cfg.max_cases_per_year:
            print(f"[{cfg.mode}] Max cases/year: {cfg.max_cases_per_year}")

        manifest = load_json(cfg.manifest_path, default={})
        failed_urls = load_json(cfg.failed_urls_path, default=[])

        print(f"[{cfg.mode}] Loaded manifest records: {len(manifest)}")
        print(f"[{cfg.mode}] Loaded failed URLs     : {len(failed_urls)}")

        all_failed_urls = list(failed_urls)

        for year in range(cfg.start_year, cfg.end_year + 1):
            print(f"\n=== {cfg.mode} {year} ===")
            year_dir = cfg.root_dir / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)

            index_url = f"{BAILII_BASE_URL}{cfg.base_path}{year}/"
            index_html = self.fetch(index_url)

            if index_html == "NOT_FOUND":
                print(f"[{cfg.mode}] {year}: year index does not exist, skipping")
                continue

            if not index_html:
                print(f"[{cfg.mode}] {year}: failed to fetch year index")
                all_failed_urls.append(index_url)
                self.checkpoint(manifest, sorted(set(all_failed_urls)))
                continue

            case_urls = self.extract_case_links(index_html, year)
            print(f"[{cfg.mode}] {year}: found {len(case_urls)} candidate case URLs")

            if cfg.max_cases_per_year is not None:
                original_count = len(case_urls)
                case_urls = case_urls[: cfg.max_cases_per_year]
                print(
                    f"[{cfg.mode}] {year}: smoke cap selected "
                    f"{len(case_urls)}/{original_count} case URLs"
                )

            if not case_urls:
                print(f"[{cfg.mode}] {year}: no case URLs found")
                continue

            results = {"ok": 0, "skip": 0, "fail": 0, "error": 0}
            failed_urls_for_year: List[str] = []
            completed_since_checkpoint = 0

            with ThreadPoolExecutor(max_workers=cfg.max_threads) as executor:
                futures = [
                    executor.submit(self.process_case, url, year_dir)
                    for url in case_urls
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{cfg.mode} {year}",
                    unit="case",
                ):
                    status, payload = future.result()
                    results[status] += 1
                    completed_since_checkpoint += 1

                    if status == "ok":
                        with self.manifest_lock:
                            manifest[payload["case_id"]] = payload
                    elif status in {"fail", "error"}:
                        failed_url = payload["url"]
                        failed_urls_for_year.append(failed_url)
                        all_failed_urls.append(failed_url)

                    self.maybe_throttle(results)

                    if completed_since_checkpoint >= cfg.checkpoint_every:
                        self.checkpoint(manifest, sorted(set(all_failed_urls)))
                        completed_since_checkpoint = 0

            self.checkpoint(manifest, sorted(set(all_failed_urls)))
            print(f"[{cfg.mode}] {year}: first pass results -> {results}")

            retry_results = self.retry_failed_cases(
                year,
                year_dir,
                failed_urls_for_year,
                manifest,
            )
            self.checkpoint(manifest, sorted(set(all_failed_urls)))
            print(f"[{cfg.mode}] {year}: retry results      -> {retry_results}")

        final_failed = self.unresolved_failures(all_failed_urls, manifest)
        self.checkpoint(manifest, final_failed)

        print(f"\n[{cfg.mode}] ALL DONE")
        print(f"[{cfg.mode}] Final manifest records: {len(manifest)}")
        print(f"[{cfg.mode}] Final failed URLs     : {len(final_failed)}")

        return {"manifest": manifest, "failed_urls": final_failed}

    def get_session(self) -> requests.Session:
        session = getattr(self.thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": self.config.user_agent})
            self.thread_local.session = session
        return session

    def fetch(self, url: str) -> Optional[str]:
        for attempt in range(1, self.config.max_retries + 1):
            try:
                with self.adaptive_sleep_lock:
                    sleep_min = self.adaptive_sleep_min
                    sleep_max = self.adaptive_sleep_max

                time.sleep(random.uniform(sleep_min, sleep_max))
                response = self.get_session().get(url, timeout=self.config.timeout)

                if response.status_code == 404:
                    return "NOT_FOUND"
                if response.status_code == 200:
                    return response.text
                if response.status_code in RETRY_STATUSES:
                    time.sleep(0.6 * attempt)
                    continue

                return None
            except Exception:
                time.sleep(0.6 * attempt)

        return None

    def extract_case_links(self, index_html: str, year: int) -> List[str]:
        soup = BeautifulSoup(index_html, "html.parser")
        links = set()

        for link in soup.find_all("a"):
            href = link.get("href")
            if not href:
                continue

            full_url = normalize_url(urljoin(BAILII_BASE_URL, href))
            if self.is_valid_case_url(full_url, year):
                links.add(full_url)

        return sorted(links)

    def is_valid_case_url(self, url: str, year: int) -> bool:
        url = normalize_url(url)
        return (
            url.startswith(BAILII_BASE_URL)
            and f"{self.config.base_path}{year}/" in url
            and url.endswith(".html")
            and "?" not in url
            and "#" not in url
        )

    def checkpoint(self, manifest: Dict[str, Any], failed_urls: List[str]) -> None:
        with self.manifest_lock:
            write_json_atomic(self.config.manifest_path, manifest)

        with self.failures_lock:
            write_json_atomic(self.config.failed_urls_path, failed_urls)

    def maybe_throttle(self, results_counter: Dict[str, int]) -> None:
        total = sum(results_counter.values())
        if total == 0:
            return

        bad = results_counter.get("fail", 0) + results_counter.get("error", 0)
        error_rate = bad / total

        with self.adaptive_sleep_lock:
            if error_rate >= self.config.error_rate_threshold:
                self.adaptive_sleep_min = min(
                    self.adaptive_sleep_min * self.config.throttle_multiplier,
                    self.config.max_sleep_cap,
                )
                self.adaptive_sleep_max = min(
                    self.adaptive_sleep_max * self.config.throttle_multiplier,
                    self.config.max_sleep_cap,
                )
            else:
                self.adaptive_sleep_min = max(
                    self.config.sleep_min,
                    self.adaptive_sleep_min / 1.10,
                )
                self.adaptive_sleep_max = max(
                    self.config.sleep_max,
                    self.adaptive_sleep_max / 1.10,
                )

    def process_case(self, url: str, year_dir: Path) -> Tuple[str, Dict[str, Any]]:
        case_id = extract_case_id(url)
        raw_path = year_dir / f"{case_id}.html"
        text_path = year_dir / f"{case_id}.txt"

        if raw_path.exists() and text_path.exists():
            return (
                "skip",
                {
                    "case_id": case_id,
                    "url": url,
                    "raw_path": str(raw_path),
                    "text_path": str(text_path),
                },
            )

        html = self.fetch(url)

        if html == "NOT_FOUND":
            return ("fail", {"case_id": case_id, "url": url, "reason": "404"})

        if not html:
            return ("fail", {"case_id": case_id, "url": url, "reason": "fetch_failed"})

        try:
            raw_path.write_text(html, encoding="utf-8")
            clean_text = clean_html_to_text(html)
            text_path.write_text(clean_text, encoding="utf-8")

            title = ""
            try:
                soup = BeautifulSoup(html, "html.parser")
                title = (soup.title.string or "").strip() if soup.title else ""
            except Exception:
                title = ""

            return (
                "ok",
                {
                    "case_id": case_id,
                    "url": url,
                    "title": title,
                    "raw_path": str(raw_path),
                    "text_path": str(text_path),
                    "year": year_dir.name,
                    "source": "BAILII",
                    "mode": self.config.mode,
                },
            )
        except Exception as exc:
            return ("error", {"case_id": case_id, "url": url, "reason": repr(exc)})

    def retry_failed_cases(
        self,
        year: int,
        year_dir: Path,
        failed_urls_for_year: List[str],
        manifest: Dict[str, Any],
    ) -> Dict[str, int]:
        if not failed_urls_for_year:
            return {"ok": 0, "skip": 0, "fail": 0, "error": 0}

        results = {"ok": 0, "skip": 0, "fail": 0, "error": 0}
        retry_candidates = []

        for url in failed_urls_for_year:
            case_id = extract_case_id(url)
            record = manifest.get(case_id)
            if record and files_exist(record):
                continue
            retry_candidates.append(url)

        if not retry_candidates:
            return results

        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            futures = [
                executor.submit(self.process_case, url, year_dir)
                for url in retry_candidates
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{self.config.mode} {year} retry",
                unit="case",
            ):
                status, payload = future.result()
                results[status] += 1

                if status == "ok":
                    with self.manifest_lock:
                        manifest[payload["case_id"]] = payload

        return results

    def unresolved_failures(
        self,
        failed_urls: List[str],
        manifest: Dict[str, Any],
    ) -> List[str]:
        final_failed = []
        for url in sorted(set(failed_urls)):
            case_id = extract_case_id(url)
            record = manifest.get(case_id)
            if record and files_exist(record):
                continue
            final_failed.append(url)
        return final_failed


def normalize_url(url: str) -> str:
    return url.strip()


def extract_case_id(url: str) -> str:
    return normalize_url(url).split("/")[-1].replace(".html", "")


def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
    return "\n".join(line for line in lines if line)


def files_exist(record: Dict[str, Any]) -> bool:
    raw_ok = Path(record.get("raw_path", "")).exists()
    text_ok = Path(record.get("text_path", "")).exists()
    return raw_ok and text_ok


def run_bailii_download(**kwargs: Any) -> Dict[str, Any]:
    return BailiiDownloader(BailiiConfig(**kwargs)).run()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download BAILII ET/EAT case HTML and text.")
    parser.add_argument("--mode", choices=sorted(BAILII_PATHS), default="ET")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--root-dir", type=Path, default=None)
    parser.add_argument("--max-threads", type=int, default=None)
    parser.add_argument("--max-cases-per-year", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one thread and process only a few cases per year.",
    )
    parser.add_argument("--sleep-min", type=float, default=None)
    parser.add_argument("--sleep-max", type=float, default=None)
    args = parser.parse_args(argv)

    max_cases_per_year = args.max_cases_per_year
    if args.smoke_test and max_cases_per_year is None:
        max_cases_per_year = 3

    run_bailii_download(
        mode=args.mode,
        start_year=args.start_year,
        end_year=args.end_year,
        root_dir=args.root_dir,
        max_threads=args.max_threads or (1 if args.smoke_test else 20),
        max_cases_per_year=max_cases_per_year,
        sleep_min=(
            args.sleep_min
            if args.sleep_min is not None
            else (5.0 if args.smoke_test else 0.05)
        ),
        sleep_max=(
            args.sleep_max
            if args.sleep_max is not None
            else (10.0 if args.smoke_test else 0.20)
        ),
    )


if __name__ == "__main__":
    main()
