#!/usr/bin/env python3
"""Extract files from a published container image without Docker.

Some projects bake build outputs into their published image (compiled
dashboard bundles, vendored JS, ONNX models) that a from-source checkout does
not carry. This pulls those layers straight out of the registry using the
anonymous token flow, so they can be copied next to a source checkout.

Prints a SHA-256 per extracted file so the result can be checked against the
pins in the image's own Dockerfile.

Usage:
    python3 ghcr-extract.py --repo owner/name --match models/ \\
        --match static/dashboard.js --out ./controller
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request

ACCEPT = ",".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def _request(url: str, headers: dict[str, str] | None = None, raw: bool = False):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    return data if raw else json.loads(data)


def _token(repo: str) -> str:
    url = (
        "https://ghcr.io/token?scope=repository:"
        f"{repo}:pull&service=ghcr.io"
    )
    return _request(url)["token"]


def _platform_digest(index: dict, arch: str, os_name: str) -> str:
    for entry in index.get("manifests", []):
        platform = entry.get("platform", {})
        if (
            platform.get("architecture") == arch
            and platform.get("os") == os_name
        ):
            return entry["digest"]
    raise SystemExit(f"no {os_name}/{arch} manifest in the image index")


def _layers(repo: str, digest: str, token: str) -> list[str]:
    headers = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}
    manifest = _request(
        f"https://ghcr.io/v2/{repo}/manifests/{digest}", headers=headers
    )
    return [layer["digest"] for layer in manifest.get("layers", [])]


def _extract_layer(
    repo: str, digest: str, token: str, matches: list[str], out: str, prefix: str
) -> list[tuple[str, int, str]]:
    headers = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}
    blob = _request(
        f"https://ghcr.io/v2/{repo}/blobs/{digest}", headers=headers, raw=True
    )
    try:
        tar = tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob)))
    except (OSError, gzip.BadGzipFile, tarfile.TarError):
        return []

    written: list[tuple[str, int, str]] = []
    for member in tar.getmembers():
        if not member.isfile():
            continue
        name = member.name
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        if not any(name.startswith(m) for m in matches):
            continue
        data = tar.extractfile(member).read()
        dest = os.path.join(out, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as handle:
            handle.write(data)
        written.append((name, len(data), hashlib.sha256(data).hexdigest()))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. owner/name")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--arch", default="amd64")
    parser.add_argument("--os", dest="os_name", default="linux")
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="path prefix to extract (repeatable)",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--prefix",
        default="app/",
        help="leading path inside the image to strip (empty for none)",
    )
    args = parser.parse_args()

    if not args.match:
        parser.error("at least one --match is required")

    token = _token(args.repo)
    headers = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}
    index = _request(
        f"https://ghcr.io/v2/{args.repo}/manifests/{args.tag}", headers=headers
    )
    if "manifests" in index:
        digest = _platform_digest(index, args.arch, args.os_name)
    else:
        digest = args.tag

    written: list[tuple[str, int, str]] = []
    for layer in _layers(args.repo, digest, token):
        written.extend(
            _extract_layer(
                args.repo, layer, token, args.match, args.out, args.prefix
            )
        )

    if not written:
        print("nothing matched", file=sys.stderr)
        return 1

    total = 0
    for name, size, digest_hex in sorted(written):
        print(f"{name}  {size}B  sha256:{digest_hex}")
        total += size
    print(f"\n{len(written)} files, {total // 1024} KB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
