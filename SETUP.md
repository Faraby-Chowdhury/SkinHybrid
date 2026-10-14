# SETUP — data acquisition and running the project

Everything here comes from **official sources**: the ISIC Challenge S3 bucket and
Harvard Dataverse. No Kaggle account, no login, no scraping.

---

## Part 1 — Two things on those pages change the project

### 1.1 `lesion_id` is in the official ISIC 2019 metadata

`ISIC_2019_Training_Metadata.csv` has five columns:

```
image,age_approx,anatom_site_general,lesion_id,sex
ISIC_0012756_downsampled,55,anterior torso,MSK4_0010029,male
ISIC_0013184_downsampled,55,anterior torso,MSK4_0010029,male
ISIC_0013874_downsampled,60,anterior torso,MSK4_0010029,male
ISIC_0014289_downsampled,60,anterior torso,MSK4_0010029,male
```

Four rows, four different image files, **one physical lesion**. A plain
`train_test_split` scatters those four across train, validation and test. The model
then "generalises" to a photograph of a mole it already memorised.

Coverage is partial (mostly the MSK portion; blank for the early ISIC and BCN rows),
so the notebook stacks three grouping sources:

1. official ISIC `lesion_id` — authoritative
2. HAM10000 `lesion_id` from Dataverse — covers the HAM portion
3. perceptual-hash union-find — catches the rest (BCN20000)

### 1.2 The official ISIC 2019 test set now has public ground truth

| | |
|---|---|
| `ISIC_2019_Test_Input.zip` | 3.6 GB, 8,238 images |
| `ISIC_2019_Test_GroundTruth.csv` | 454 KB, includes the official scoring weights |

This is the set **Gessert et al. won on with 63.6% balanced accuracy**. You can score
against the real leaderboard instead of your own split. That is the difference between
"our held-out split gave 0.58" and "we scored 0.58 on the same benchmark the winner
scored 0.636 on" — the second sentence is a paper, the first is a homework result.

> **Caveat to state in the paper:** the official test set contains a ninth
> out-of-distribution "unknown" class absent from training. Restrict scoring to the
> eight known classes and say so, or your numbers will not line up with the leaderboard.

### 1.3 Note on `_downsampled`

Many image IDs carry a `_downsampled` suffix (`ISIC_0012756_downsampled`). The ground
truth, metadata and filenames all use the suffixed form consistently, so joins work —
but do not strip it, or you will silently drop thousands of rows.

---

## Part 2 — Download

```bash
mkdir dermaudit && cd dermaudit
# put DermAudit.ipynb, download_data.py, requirements.txt here

python download_data.py --root ./data              # ~13 GB, includes official test set
python download_data.py --root ./data --core-only  # ~9 GB, skips the 3.6 GB test images
python download_data.py --root ./data --verify     # check what landed
```

The script is **resumable** — interrupt it, run it again, it continues from where it
stopped. It skips anything already complete.

For Harvard Dataverse it queries the dataset API for current file IDs rather than
hard-coding them, because Dataverse reassigns IDs between versions and a stale ID
fails silently.

**Disk:** allow ~30 GB. The zips are kept alongside their extracted contents; delete
the `.zip` files after `--verify` passes to reclaim about 13 GB.

**Time:** 1-3 hours on a typical connection. Start it before you go to bed.

### Manual fallback

If your network blocks the script, download by hand:

**ISIC** — https://challenge.isic-archive.com/data/ → the **2019** tab:
- Training Data (9.1 GB) · Training Ground Truth (1 MB) · Task 2 Metadata (1 MB)
- Test Data (3.6 GB) · Test Ground Truth (454 KB)

**HAM10000** — https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- `HAM10000_segmentations_lesion_tschandl.zip` (10.3 MB) — the masks
- `HAM10000_metadata.tab` (811 KB) — the lesion IDs

You do **not** need `HAM10000_images_part_1/2.zip` (2.6 GB). Those images are already
inside ISIC 2019.

**PH2** (optional) — https://www.fc.up.pt/addi/ph2%20database.html, free registration.

---

## Part 3 — Directory layout

Target state after download:

```
dermaudit/
├── DermAudit.ipynb
├── download_data.py
├── requirements.txt
├── ROADMAP.md
├── SETUP.md
│
├── data/
│   ├── ISIC_2019_Training_Input/         25,331 .jpg
│   ├── ISIC_2019_Training_GroundTruth.csv
│   ├── ISIC_2019_Training_Metadata.csv   <- lesion_id lives here
│   ├── ISIC_2019_Test_Input/              8,238 .jpg
│   ├── ISIC_2019_Test_GroundTruth.csv
│   ├── HAM10000_metadata.csv             <- converted from .tab
│   ├── HAM10000_segmentations/           10,015 .png
│   ├── PH2Dataset/                        optional
│   └── cache_256/                         built by the notebook, ~1.2 GB
│
└── outputs/                               created by the notebook
    ├── figures/    F1..F5 .png
    ├── tables/     T1..T7 .csv
    ├── checkpoints/  *.pt
    ├── predictions/  *.npz
    └── summary.json
    results.zip                            final deliverable
```

`--verify` should end with:

```
OK   ISIC 2019 training images        25331 files (expected 25331)
OK   ISIC 2019 test images             8238 files (expected 8238)
OK   HAM10000 masks                   10015 files (expected 10015)

lesion_id populated on N/25331 rows
distinct lesions among those: M
=> at least N-M images share a lesion with another image.
   These are exactly what an image-level split leaks.
```

**That last number is your first result.** Write it down before training anything.

---

## Part 4 — Environment

### Local, RTX 3060 Ti, VS Code (recommended)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name dermaudit
```

Check before anything else:
```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
If that prints `False`, stop and fix CUDA. Everything downstream assumes a GPU.

In VS Code: install the **Python** and **Jupyter** extensions, open `DermAudit.ipynb`,
pick the `dermaudit` kernel top-right.

**Windows note:** `CFG.NUM_WORKERS` is already forced to 0 on Windows. PyTorch's
spawn-based workers deadlock inside notebooks there. Leave it.

### Colab (fallback)

Colab is worse for this job — it disconnects during the 9-hour grid. If you must:
mount Drive and set `CFG.DATA_ROOT` and `CFG.OUT_ROOT` to Drive paths so the cache and
checkpoints survive. The run loop skips completed runs, so reconnect-and-rerun works.

---

## Part 5 — Running order

| Step | Cells | Time | Watch for |
|---|---|---|---|
| 1 | Config + imports | seconds | `torch.cuda.is_available()` is True |
| 2 | `check_data()` | seconds | required files all OK |
| 3 | Build manifest | ~1 min | lesion_id coverage percentage |
| 4 | **Build 256px cache** | 10-20 min | do not skip — biggest speedup in the project |
| 5 | pHash + grouping | 5-10 min | images vs distinct groups |
| 6 | Splits + **T1 leakage** | seconds | **lesion split must show exactly 0 leaked** |
| 7 | The grid, 18 runs | ~9 h | interrupt-safe, resume anytime |
| 8 | Analysis + figures | ~5 min | T2 AUC spread vs bAcc spread |
| 9 | AMIL audit | ~10 min | needs the HAM masks |
| 10 | Package | seconds | `results.zip` |

**Stop at step 6 and read T1.** If the lesion split shows anything other than zero
leaked images, the grouping is broken and nothing after it is trustworthy.

---

## Part 6 — Before you train anything: check the masks

Twenty minutes, and it may change what the paper says.

Your old code ran `THRESH_BINARY_INV + THRESH_OTSU` on the **whole greyscale frame**.
ISIC images often have dark vignetting from the dermoscope aperture. Inverse Otsu will
happily select those corners as foreground, making the largest contour the vignette
rather than the lesion.

That hypothesis predicts exactly what your firefly wrapper found: it discarded every
global shape descriptor (compactness, eccentricity, ellipse fit — all corrupted by a
wrong contour) and kept colour statistics (roughly stable even inside a wrong mask).

```python
import cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
paths = sorted(Path("data/ISIC_2019_Training_Input").glob("*.jpg"))[:50]
fig, axes = plt.subplots(5, 10, figsize=(20, 10))
for ax, p in zip(axes.ravel(), paths):
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    g  = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ax.imshow(im); ax.contour(m, [127], colors="lime", linewidths=0.8); ax.axis("off")
plt.tight_layout(); plt.savefig("mask_check.png", dpi=110, bbox_inches="tight")
```

Green outlines on corners instead of lesions is a **finding**, not just a bug — it is
the reason handcrafted shape features underperform in hybrid pipelines generally.

---

## Part 7 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | batch too large for 8 GB | `CFG.BATCH_SIZE = 32` |
| Epochs take >5 min | cache not built | re-run the cache cell; confirm `cache_256/` has 25,331 files |
| GPU sits near 0% | data loading bound | build the cache; raise `NUM_WORKERS` (non-Windows) |
| Lesion split shows leaked > 0 | grouping broken | check lesion_id coverage in step 3 |
| Notebook hangs on first batch (Windows) | worker spawn | `CFG.NUM_WORKERS = 0` |
| `pyarrow` error | old notebook version | already fixed; hash cache is CSV now |
| Dataverse download fails | API/network | use the manual links in Part 2 |
| Manifest has far fewer than 25,331 rows | `_downsampled` suffix stripped | do not modify image IDs |

---

## Part 8 — Citations you owe

ISIC 2019 is **CC-BY-NC**. Non-commercial use only, attribution required. Cite all
three constituent sources plus the aggregate:

> BCN_20000: © Department of Dermatology, Hospital Clínic de Barcelona.
> HAM10000: © ViDIR Group, Department of Dermatology, Medical University of Vienna,
> doi:10.1038/sdata.2018.161. MSK: © Anonymous, arXiv:1710.05006, arXiv:1902.03368.

Full references:
1. Tschandl P., Rosendahl C., Kittler H. *Sci. Data* **5**, 180161 (2018).
2. Codella N. et al. arXiv:1710.05006 (2017).
3. Hernández-Pérez C. et al. *Sci. Data* **11**(1), 641 (2024).

For the segmentation masks, additionally cite:
4. Tschandl P. et al. Human-computer collaboration for skin cancer recognition.
   *Nature Medicine* (2020). doi:10.1038/s41591-020-0942-0

For the baseline you compare against:
5. Gessert N. et al. Skin lesion classification using ensembles of multi-resolution
   EfficientNets with meta data. *MethodsX* **7**, 100864 (2020). — 63.6% balanced
   accuracy, first place, ISIC 2019 Task 1.
