#!/usr/bin/env python3
import argparse
import os
import re
import time
import sys
import urllib.parse
import urllib.request


INDEX_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench/"


def list_npz_files(index_url):
    with urllib.request.urlopen(index_url) as response:
        html = response.read().decode("utf-8", errors="replace")
    names = []
    for href in re.findall(r'href="([^"]+\.npz)"', html):
        name = urllib.parse.unquote(href)
        if "/" not in name:
            names.append(name)
    return sorted(set(names))


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length is not None else None


def download_file(url, path):
    tmp_path = f"{path}.tmp"
    resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0

    request = urllib.request.Request(url)
    if resume_from > 0:
        request.add_header("Range", f"bytes={resume_from}-")

    with urllib.request.urlopen(request) as response:
        if resume_from > 0 and response.status != 206:
            resume_from = 0

        total = response.headers.get("Content-Length")
        total = int(total) if total is not None else None
        if response.status == 206:
            content_range = response.headers.get("Content-Range")
            if content_range is not None:
                total = int(content_range.rsplit("/", 1)[1])

        mode = "ab" if resume_from > 0 else "wb"
        downloaded = resume_from
        last_report = resume_from
        if total is not None and downloaded > 0:
            pct = 100.0 * downloaded / total
            print(f"  resuming at {downloaded / 1024**3:.2f} / {total / 1024**3:.2f} GiB ({pct:.1f}%)", flush=True)

        with open(tmp_path, mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total is not None and downloaded - last_report >= 64 * 1024 * 1024:
                    pct = 100.0 * downloaded / total
                    print(
                        f"  {downloaded / 1024**3:.2f} / {total / 1024**3:.2f} GiB ({pct:.1f}%)",
                        flush=True,
                    )
                    last_report = downloaded

    if total is not None and os.path.getsize(tmp_path) != total:
        raise RuntimeError(f"Incomplete download for {path}: got {os.path.getsize(tmp_path)} bytes, expected {total}")

    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default=os.environ.get(
            'OGBENCH_DATASET_DIR',
            os.path.join(
                os.environ.get('DATA_ROOT', '/data/qijunrong/06-RL/offline-rl'),
                'data',
                'raw_ogbench',
            ),
        ),
        help="Directory where OGBench .npz files are stored.",
    )
    parser.add_argument("--index_url", default=INDEX_URL)
    parser.add_argument("--include_visual", action="store_true", help="Also download visual datasets.")
    parser.add_argument("--include_regex", default=None, help="Only download file names matching this regex.")
    parser.add_argument("--exclude_regex", default=None, help="Skip file names matching this regex.")
    parser.add_argument("--retries", type=int, default=5, help="Retries per file before giving up.")
    parser.add_argument("--retry_sleep", type=float, default=10.0, help="Seconds to sleep between retries.")
    parser.add_argument("--dry_run", action="store_true", help="Only print missing files.")
    args = parser.parse_args()

    os.makedirs(args.dataset_dir, exist_ok=True)
    names = list_npz_files(args.index_url)
    if not args.include_visual:
        names = [name for name in names if not name.startswith("visual-")]
    if args.include_regex is not None:
        include_re = re.compile(args.include_regex)
        names = [name for name in names if include_re.search(name)]
    if args.exclude_regex is not None:
        exclude_re = re.compile(args.exclude_regex)
        names = [name for name in names if not exclude_re.search(name)]

    missing_or_incomplete = []
    for name in names:
        url = urllib.parse.urljoin(args.index_url, name)
        path = os.path.join(args.dataset_dir, name)
        size = remote_size(url)
        if os.path.exists(path) and (size is None or os.path.getsize(path) == size):
            continue
        missing_or_incomplete.append((name, url, path, size))

    print(f"Found {len(names)} dataset files in index.")
    print(f"Need to download {len(missing_or_incomplete)} files.")
    if args.dry_run:
        for name, _, _, size in missing_or_incomplete:
            size_text = "unknown" if size is None else f"{size / 1024**3:.2f} GiB"
            print(f"{name}\t{size_text}")
        return

    for idx, (name, url, path, size) in enumerate(missing_or_incomplete, start=1):
        size_text = "unknown" if size is None else f"{size / 1024**3:.2f} GiB"
        print(f"[{idx}/{len(missing_or_incomplete)}] Downloading {name} ({size_text})", flush=True)
        for attempt in range(1, args.retries + 1):
            try:
                download_file(url, path)
                break
            except Exception as exc:
                tmp_path = f"{path}.tmp"
                if os.path.exists(tmp_path):
                    print(f"Partial file kept at {tmp_path}", file=sys.stderr, flush=True)
                if attempt >= args.retries:
                    raise
                print(
                    f"Retrying {name} after error ({attempt}/{args.retries}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(args.retry_sleep)

    print("Done.")


if __name__ == "__main__":
    main()
