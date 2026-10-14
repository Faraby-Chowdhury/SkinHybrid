#!/usr/bin/env python3
"""
download_data.py - fetch every dataset DermAudit needs from OFFICIAL sources.

Sources (no Kaggle, no login required):
  * ISIC Challenge S3      - challenge.isic-archive.com  -> isic-archive.s3.amazonaws.com
  * Harvard Dataverse API  - doi:10.7910/DVN/DBW86T (HAM10000, ViDIR / MedUni Vienna)

Usage
-----
    python download_data.py --root ./data              # core sets (~13 GB)
    python download_data.py --root ./data --all        # + ISIC2018 test, PH2 note
    python download_data.py --root ./data --core-only  # skip the 3.6 GB test images
    python download_data.py --root ./data --verify     # re-check what is present

Resumable: partial downloads continue where they stopped. Re-running skips
anything already complete.
"""
import argparse, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

S3 = "https://isic-archive.s3.amazonaws.com/challenges"
DV_API = "https://dataverse.harvard.edu/api"
HAM_DOI = "doi:10.7910/DVN/DBW86T"

# Files we want from the HAM10000 Dataverse deposit. We look their numeric IDs
# up at runtime rather than hard-coding them, because Dataverse reassigns IDs
# between dataset versions and a stale ID silently 404s.
DATAVERSE_WANTED = {
    "HAM10000_segmentations_lesion_tschandl.zip": ("HAM10000_segmentations_lesion_tschandl", True),
    "HAM10000_metadata.tab":                      ("HAM10000_metadata", False),
}


def dataverse_manifest(doi=HAM_DOI):
    """List files in a Dataverse dataset -> {filename: (id, size)}."""
    url = f"{DV_API}/datasets/:persistentId/versions/:latest/files?persistentId={doi}"
    try:
        r = urlopen(Request(url, headers={"User-Agent": "DermAudit/1.0"}), timeout=60)
        data = json.loads(r.read().decode())
    except Exception as e:
        print(f"  could not reach Dataverse API: {e}")
        return {}
    out = {}
    for entry in data.get("data", []):
        df = entry.get("dataFile", {})
        fn = df.get("filename") or entry.get("label", "")
        if fn:
            out[fn] = (df.get("id"), df.get("filesize", 0))
    return out


def fetch_dataverse(root: Path):
    """Resolve IDs at runtime, then download the HAM10000 files we need."""
    print("\n--- Harvard Dataverse: HAM10000 (doi:10.7910/DVN/DBW86T) ---")
    man = dataverse_manifest()
    if not man:
        print("  Falling back to manual instructions:")
        print("    https://dataverse.harvard.edu/dataset.xhtml?persistentId=" + HAM_DOI)
        print("    Download HAM10000_segmentations_lesion_tschandl.zip and")
        print("    HAM10000_metadata.tab into", root)
        return
    print(f"  {len(man)} files in deposit")
    for want, (stem, do_unpack) in DATAVERSE_WANTED.items():
        # match on stem so .tab/.csv/original-format variations all resolve
        hit = next((fn for fn in man if Path(fn).stem == stem or fn == want), None)
        if not hit:
            print(f"  ! {want}: not found in deposit listing")
            continue
        fid, size = man[hit]
        dest = root / want
        print(f"\n  {hit} (id {fid}, ~{human(size or 0)})")
        if dest.exists() and dest.stat().st_size > 0:
            print("    present, skipping")
        else:
            # ?format=original stops Dataverse converting tabular files
            url = f"{DV_API}/access/datafile/{fid}?format=original"
            if not download(url, dest, (size or 0) / 1e6):
                continue
        if do_unpack and dest.suffix == ".zip":
            unpack(dest, root)

# ---------------------------------------------------------------------------
# name, url, approx MB, unpack?, tier
# tier: core = required, test = official benchmark, extra = optional
# ---------------------------------------------------------------------------
FILES = [
    # ---- ISIC 2019 training corpus (the experimental corpus) ---------------
    ("ISIC_2019_Training_Input.zip",
     f"{S3}/2019/ISIC_2019_Training_Input.zip", 9100, True, "core"),
    ("ISIC_2019_Training_GroundTruth.csv",
     f"{S3}/2019/ISIC_2019_Training_GroundTruth.csv", 1, False, "core"),
    # lesion_id lives here -- this file is what makes the leakage axis possible
    ("ISIC_2019_Training_Metadata.csv",
     f"{S3}/2019/ISIC_2019_Training_Metadata.csv", 1, False, "core"),

    # ---- ISIC 2019 OFFICIAL held-out test set (ground truth now public) ----
    ("ISIC_2019_Test_Input.zip",
     f"{S3}/2019/ISIC_2019_Test_Input.zip", 3600, True, "test"),
    ("ISIC_2019_Test_GroundTruth.csv",
     f"{S3}/2019/ISIC_2019_Test_GroundTruth.csv", 1, False, "test"),
    ("ISIC_2019_Test_Metadata.csv",
     f"{S3}/2019/ISIC_2019_Test_Metadata.csv", 1, False, "test"),

    # ---- ISIC 2018 Task 3 lesion groupings (a third grouping source) -------
    ("ISIC2018_Task3_Training_LesionGroupings.csv",
     f"{S3}/2018/ISIC2018_Task3_Training_LesionGroupings.csv", 1, False, "extra"),
]


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url, dest: Path, approx_mb):
    """Resumable download with a progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0

    req = Request(url, headers={"User-Agent": "DermAudit/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"    resuming at {human(have)}")

    try:
        r = urlopen(req, timeout=60)
    except HTTPError as e:
        if e.code == 416 and have:            # already complete
            tmp.rename(dest); return True
        print(f"    HTTP {e.code} for {url}"); return False
    except URLError as e:
        print(f"    network error: {e.reason}"); return False

    total = int(r.headers.get("Content-Length", 0)) + have
    mode = "ab" if have else "wb"
    t0, done = time.time(), have
    with open(tmp, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if total:
                pct = 100 * done / total
                mbps = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r    {pct:5.1f}%  {human(done)}/{human(total)}  "
                      f"{mbps:.1f} MB/s", end="", flush=True)
    print()
    tmp.rename(dest)
    return True


def unpack(zpath: Path, root: Path):
    """Extract, flattening the common single-top-level-folder case."""
    stem = zpath.stem
    marker = root / f".unpacked_{stem}"
    if marker.exists():
        print(f"    already extracted")
        return
    print(f"    extracting {zpath.name} ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"    done")


def normalise_layout(root: Path):
    """Make the on-disk names match what the notebook expects."""
    # Dataverse serves HAM10000_metadata as .tab (tab-separated). Convert.
    tab = root / "HAM10000_metadata.tab"
    csv = root / "HAM10000_metadata.csv"
    if tab.exists() and not csv.exists():
        import csv as _csv
        with open(tab, newline="", encoding="utf-8") as fi, \
             open(csv, "w", newline="", encoding="utf-8") as fo:
            w = _csv.writer(fo)
            for row in _csv.reader(fi, delimiter="\t"):
                w.writerow(row)
        print(f"  converted HAM10000_metadata.tab -> .csv")

    # segmentation folder name varies between releases
    for cand in ["HAM10000_segmentations_lesion_tschandl",
                 "HAM10000_segmentations"]:
        p = root / cand
        if p.is_dir() and cand != "HAM10000_segmentations":
            tgt = root / "HAM10000_segmentations"
            if not tgt.exists():
                p.rename(tgt)
                print(f"  renamed {cand} -> HAM10000_segmentations")


def verify(root: Path):
    print("\n" + "=" * 66)
    print("VERIFICATION")
    print("=" * 66)
    checks = [
        ("ISIC 2019 training images", root / "ISIC_2019_Training_Input", 25331, "*.jpg"),
        ("ISIC 2019 test images",     root / "ISIC_2019_Test_Input",      8238, "*.jpg"),
        ("HAM10000 masks",            root / "HAM10000_segmentations",   10015, "*.png"),
    ]
    ok = True
    for name, path, expect, pat in checks:
        if not path.exists():
            print(f"  --   {name:32s} MISSING ({path.name})")
            continue
        n = sum(1 for _ in path.rglob(pat))
        flag = "OK  " if n == expect else "WARN"
        if n != expect and expect == 25331:
            ok = False
        print(f"  {flag} {name:32s} {n:6d} files (expected {expect})")

    for f in ["ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Metadata.csv",
              "ISIC_2019_Test_GroundTruth.csv", "HAM10000_metadata.csv"]:
        p = root / f
        print(f"  {'OK  ' if p.exists() else '--  '} {f}")

    # the single most important check: is lesion_id actually populated?
    meta = root / "ISIC_2019_Training_Metadata.csv"
    if meta.exists():
        import csv as _csv
        with open(meta, newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        if rows and "lesion_id" in rows[0]:
            n = sum(1 for r in rows if r["lesion_id"])
            uniq = len({r["lesion_id"] for r in rows if r["lesion_id"]})
            multi = n - uniq
            print(f"\n  lesion_id populated on {n}/{len(rows)} rows "
                  f"({100*n/len(rows):.1f}%)")
            print(f"  distinct lesions among those: {uniq}")
            print(f"  => at least {multi} images share a lesion with another image.")
            print("     These are exactly what an image-level split leaks.")
    print("=" * 66)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="./data", type=Path)
    ap.add_argument("--core-only", action="store_true",
                    help="skip the 3.6 GB official test images")
    ap.add_argument("--all", action="store_true", help="include optional extras")
    ap.add_argument("--verify", action="store_true", help="only verify, do not download")
    a = ap.parse_args()

    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if a.verify:
        verify(root); return

    tiers = {"core"}
    if not a.core_only:
        tiers.add("test")
    if a.all:
        tiers |= {"test", "extra"}

    plan = [f for f in FILES if f[4] in tiers]
    total_mb = sum(f[2] for f in plan)
    free = shutil.disk_usage(root).free / 1e6
    print(f"Destination : {root}")
    print(f"To download : {len(plan)} files, ~{total_mb/1000:.1f} GB")
    print(f"Disk free   : {free/1000:.1f} GB")
    if free < total_mb * 2.2:
        print("\nWARNING: you want roughly 2.2x the download size free, because the "
              "archives are extracted alongside the zips. Free some space or move --root.")
        if input("Continue anyway? [y/N] ").strip().lower() != "y":
            return

    for i, (name, url, mb, do_unpack, tier) in enumerate(plan, 1):
        dest = root / name
        print(f"\n[{i}/{len(plan)}] {name}  (~{mb} MB, {tier})")
        if dest.exists() and dest.stat().st_size > 0:
            print("    present, skipping download")
        else:
            if not download(url, dest, mb):
                print(f"    FAILED. Retry later, or fetch manually:\n      {url}")
                continue
        if do_unpack and dest.suffix == ".zip":
            unpack(dest, root)

    if "core" in tiers:
        fetch_dataverse(root)

    normalise_layout(root)
    verify(root)

    print("\nPH2 must be fetched manually (free registration, no direct link):")
    print("  https://www.fc.up.pt/addi/ph2%20database.html")
    print(f"  Unzip it to: {root / 'PH2Dataset'}")
    print("\nNext: open DermAudit.ipynb and set CFG.DATA_ROOT to")
    print(f"  {root}")


if __name__ == "__main__":
    main()
