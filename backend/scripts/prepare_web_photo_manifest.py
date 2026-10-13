"""Prepare five attributed Wikimedia photos for repeatable CPU/GPU evaluation.

Original images remain in an ignored output directory. Each original must match
the curated SHA-256; new source revisions are not silently accepted. No phone
capture or ground-truth reconstruction claim is made for these public inputs.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import ssl
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def prepare(output: Path, source_dir: Path | None = None, ca_file: str | None = None):
    records = json.loads((ROOT / "backend/app/recon/web_photo_sources.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    images = output / "originals"
    images.mkdir()
    attempts = []
    rows = []
    context = None
    for record in records:
        attempt = {"case_id": record["case_id"], "status": "FAILED", "error": None}
        attempts.append(attempt)
        try:
            name = record["file_name"]
            if Path(name).name != name:
                raise ValueError("Image filename must have no directory components")
            target = images / name
            if source_dir:
                shutil.copyfile(source_dir / name, target)
            else:
                if context is None:
                    context = ssl.create_default_context(cafile=ca_file)
                url = urlparse(record["download_url"])
                if url.scheme != "https" or url.hostname != "upload.wikimedia.org":
                    raise ValueError("Expected an HTTPS Wikimedia original image URL")
                request = Request(record["download_url"], headers={"User-Agent": "AeroVoxelResearch/1.0 (https://github.com/Yumekaz/AeroVoxel)"})
                with urlopen(request, timeout=60, context=context) as response:
                    data = response.read(20 * 1024 * 1024 + 1)
                if len(data) > 20 * 1024 * 1024:
                    raise ValueError("Image exceeds the 20 MiB input limit")
                target.write_bytes(data)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError("Source checksum changed; inspect it before updating the source record")
            with Image.open(target) as image:
                size = image.size
                image.verify()
            if size != (record["width"], record["height"]):
                raise ValueError("Unexpected source image dimensions")
            rows.append({**record, "image_path": str(target.resolve())})
            attempt.update(status="SUCCEEDED", sha256=digest)
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
        (output / "preparation.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8", newline="\n")
        print(record["case_id"], attempt["status"], attempt["error"] or "checksum verified", flush=True)
    if len(rows) != len(records):
        raise RuntimeError("Some photos failed preparation; raw attempts are preserved in preparation.json")
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "sources.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    return output / "manifest.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / f"evaluation_outputs/web_photos_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    parser.add_argument("--source-dir", type=Path, help="Reuse already downloaded originals; checksums are still verified")
    parser.add_argument("--ca-file", help="Optional trusted CA bundle for networks using a local certificate authority")
    args = parser.parse_args()
    print("Manifest:", prepare(args.output_dir.resolve(), args.source_dir, args.ca_file))
