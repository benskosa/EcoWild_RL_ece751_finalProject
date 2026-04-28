"""
scrape_hpwren_sequence.py
-------------------------
Downloads a contiguous range of HPWREN timestamp-named JPEGs from a
FIgLib long-duration index page.

Usage
-----
    # Smoke sequence (1591645419 to 1591672899):
    python scrape_hpwren_sequence.py \
        --index_url https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Miscellaneous/HPWREN-FIgLib-LongDurations/20200608-borderarea-om-s-mobo-c/index.html \
        --start 1591645419 \
        --end   1591672899 \
        --out_dir energy_eval_sequences/smoke

    # No-smoke sequence (pick a different 3-hour window from the same or different index):
    python scrape_hpwren_sequence.py \
        --index_url <url> \
        --start <timestamp> \
        --end   <timestamp> \
        --out_dir energy_eval_sequences/no_smoke
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_image_urls(index_url: str, start: int, end: int) -> list[tuple[int, str]]:
    """
    Fetch the index page and return (timestamp, image_url) pairs where
    start <= timestamp <= end, in ascending order.
    """
    print(f"Fetching index: {index_url}")
    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    base_url = index_url.rsplit("/", 1)[0] + "/"

    results = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if not href.lower().endswith(".jpg"):
            continue
        stem = Path(href).stem
        try:
            ts = int(stem)
        except ValueError:
            continue
        if start <= ts <= end:
            full_url = urljoin(base_url, href)
            results.append((ts, full_url))

    results.sort(key=lambda x: x[0])
    return results


def download_images(
    image_urls: list[tuple[int, str]],
    out_dir: Path,
    delay_s: float = 0.2,
    retries: int = 3,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(image_urls)} images to: {out_dir}")

    failed = []
    for ts, url in tqdm(image_urls, unit="img"):
        dest = out_dir / f"{ts}.jpg"
        if dest.exists():
            continue  # already downloaded

        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                dest.write_bytes(r.content)
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"\n  FAILED ({url}): {e}")
                    failed.append(url)
                else:
                    time.sleep(1.0)

        time.sleep(delay_s)

    print(f"\nDone. {len(image_urls) - len(failed)} downloaded, {len(failed)} failed.")
    if failed:
        print("Failed URLs:")
        for u in failed:
            print(f"  {u}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--index_url", required=True,
                        help="URL of the HPWREN FIgLib index.html page")
    parser.add_argument("--start", type=int, required=True,
                        help="First Unix timestamp to download (inclusive)")
    parser.add_argument("--end",   type=int, required=True,
                        help="Last Unix timestamp to download (inclusive)")
    parser.add_argument("--out_dir", default="energy_eval_sequences/smoke",
                        help="Directory to save downloaded images")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Seconds to wait between requests (default: 0.2)")
    args = parser.parse_args()

    image_urls = get_image_urls(args.index_url, args.start, args.end)

    if not image_urls:
        print(f"No images found in range [{args.start}, {args.end}]. "
              f"Check --start/--end against the index page.")
        return

    print(f"Found {len(image_urls)} images in range "
          f"[{image_urls[0][0]}, {image_urls[-1][0]}]")

    download_images(image_urls, Path(args.out_dir), delay_s=args.delay)


if __name__ == "__main__":
    main()
