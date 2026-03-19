"""
HPWREN Image Scraper
--------------------
Scrapes all images from the HPWREN FIgLib index page and saves them
to a folder of your choice.

Requirements:
    pip install requests beautifulsoup4

Usage:
    python scrape_hpwren_images.py

    Or specify a custom output folder:
    python scrape_hpwren_images.py --output ~/Desktop/MyFireImages
"""

import os
import time
import argparse
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_URL = (
    "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/"
    "20160604_FIRE_rm-n-mobo-c/index.html"
)

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Desktop", "HPWREN_Images")

# Image extensions to look for
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# Seconds to wait between downloads (be polite to the server)
DOWNLOAD_DELAY = 0.3

# ── Helpers ──────────────────────────────────────────────────────────────────

def is_image_url(url: str) -> bool:
    """Return True if the URL points to a recognised image file."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS)


def fetch_image_urls(page_url: str) -> list[str]:
    """
    Parse the HTML page and return a list of absolute image URLs found in
    <img> tags and any <a> links that point directly to image files.
    """
    print(f"Fetching page: {page_url}")
    resp = requests.get(page_url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    found: set[str] = set()

    # <img src="...">
    for tag in soup.find_all("img"):
        src = tag.get("src") or tag.get("data-src")
        if src:
            found.add(urljoin(page_url, src))

    # <a href="..."> links that point to image files
    for tag in soup.find_all("a", href=True):
        href = urljoin(page_url, tag["href"])
        if is_image_url(href):
            found.add(href)

    image_urls = [u for u in found if is_image_url(u)]
    print(f"Found {len(image_urls)} image(s).")
    return sorted(image_urls)


def download_images(image_urls: list[str], output_dir: str) -> None:
    """Download each image URL into output_dir, skipping files that exist."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}\n")

    total = len(image_urls)
    success = skipped = failed = 0

    for i, url in enumerate(image_urls, start=1):
        filename = os.path.basename(urlparse(url).path) or f"image_{i}"
        dest = os.path.join(output_dir, filename)

        if os.path.exists(dest):
            print(f"[{i}/{total}] Skipping (already exists): {filename}")
            skipped += 1
            continue

        try:
            resp = requests.get(url, timeout=30, stream=True)
            resp.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = os.path.getsize(dest) / 1024
            print(f"[{i}/{total}] Downloaded ({size_kb:,.1f} KB): {filename}")
            success += 1

        except Exception as exc:
            print(f"[{i}/{total}] FAILED: {filename} — {exc}")
            failed += 1

        time.sleep(DOWNLOAD_DELAY)

    print(f"\n── Summary ─────────────────────────────")
    print(f"  Downloaded : {success}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {failed}")
    print(f"  Location   : {output_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape images from the HPWREN FIgLib page.")
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Destination folder (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--url", "-u",
        default=TARGET_URL,
        help="Override the target page URL.",
    )
    args = parser.parse_args()

    image_urls = fetch_image_urls(args.url)

    if not image_urls:
        print("No images found. The page structure may have changed.")
        return

    download_images(image_urls, args.output)


if __name__ == "__main__":
    main()