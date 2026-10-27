# CCA 3 – Computer Vision Practical Assignment (Units 3, 4 and 5)

Python implementation of image restoration, segmentation, feature extraction,
object recognition and motion estimation, applied to the student's own
photograph and (once supplied) a self-recorded video.

Libraries: **OpenCV, NumPy, Matplotlib, scikit-image, Ultralytics YOLO**.

| Task | Unit | What is implemented | Code |
|---|---|---|---|
| 1A | 3 | Gaussian and salt-and-pepper noise, Gaussian and median filters (3–11 px), PSNR and SSIM | `src/image_restoration.py` |
| 1B | 3 | Grayscale, Otsu and adaptive thresholding, erosion, dilation, opening, closing, Canny, watershed | `src/segmentation.py` |
| 2 | 4 | HOG, SIFT keypoints and descriptors, BF/FLANN matching with ratio test and RANSAC, rotation / resize / lighting experiments | `src/feature_extraction.py` |
| 3 | 5 | Viola–Jones Haar cascade (face + eyes) and YOLO11n; minNeighbors sweep and rotation robustness | `src/object_detection.py` |
| 4 | 5 | Frame differencing, GMM background subtraction (MOG2), Lucas–Kanade optical flow, output video and GIF | `src/motion_estimation.py` |

## Current status

| Input | Status | Effect |
|---|---|---|
| `inputs/original_photo.jpg` | Present | Units 3, 4 and 5A complete |
| `inputs/second_photo.jpg` | **Missing** | Two-photo SIFT matching pending (a clearly labelled synthetic view is used as a preliminary test) |
| `inputs/original_video.mp4` | **Missing** | Unit 5B motion results and detections on video frames pending (code complete and tested on a synthetic clip) |

See [`inputs/README.md`](inputs/README.md) for what to capture, then run
`python main.py --all` and `python tools/build_report.py` again.

## Folder structure

```
StudentComputerVisionAssignment/
├── main.py                  # entry point: --unit 3/4/5, --all, --check-inputs
├── requirements.txt         # packages (requirements-lock.txt = exact tested versions)
├── src/
│   ├── config.py            # every parameter and file name in one place
│   ├── utils.py             # input checks, image I/O, figures, JSON/CSV, timer
│   ├── image_restoration.py # Unit 3A
│   ├── segmentation.py      # Unit 3B
│   ├── feature_extraction.py# Unit 4
│   ├── object_detection.py  # Unit 5A
│   └── motion_estimation.py # Unit 5B
├── inputs/                  # your photo / video (kept out of Git)
├── outputs/
│   ├── unit3/restoration/   # noisy + restored images, metrics CSV, figures
│   ├── unit3/segmentation/  # masks, morphology, edges, watershed
│   ├── unit4/               # HOG, SIFT keypoints, matches, sweep tables
│   ├── unit5/detection/     # Haar / YOLO images and tables
│   ├── unit5/motion/        # (after the video is added) masks, vectors, video, GIF
│   └── run_summary.json     # status of every task in the last run
├── reports/                 # CCA3_Technical_Report.docx / .pdf
├── docs/                    # algorithm explanations, video script, submission checklist
├── tests/                   # pytest suite (synthetic + public sample data only)
├── tools/                   # build_report.py, export_pdf.py
├── notebooks/               # how to use the code from Jupyter
└── models/                  # YOLO weights, downloaded automatically (not in Git)
```

## Setup on Windows

Requires Python 3.10–3.12 (tested with 3.11.18) and about 1.5 GB of disk space
for PyTorch. In PowerShell, from the project folder:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run `activate`, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call the interpreter
directly: `.venv\Scripts\python.exe main.py --all`.

> **Note for Anaconda users:** a venv made from Anaconda's own `python.exe` can
> load an old Visual C++ runtime and PyTorch then fails with *WinError 1114
> (c10.dll)*. Use a python.org or conda-forge Python for the venv. This project
> was built with a conda-forge Python 3.11 kept in `.tools/python311`
> (`conda create -p .tools\python311 -c conda-forge python=3.11`, then
> `.tools\python311\python.exe -m venv .venv`).

## Running

```
python main.py --check-inputs     # which inputs are present
python main.py --unit 3           # restoration + segmentation
python main.py --unit 4           # HOG + SIFT
python main.py --unit 5           # Haar/YOLO + motion
python main.py --all              # everything
python main.py --unit 5 --skip-yolo
python main.py --all --show       # also open every figure in a window
```

Options: `--inputs`, `--outputs`, `--models` change the folders;
`--yolo-model yolov8n.pt` picks other official weights; `--verbose` prints more.

A missing photo or video never stops the run: the task is reported as
**PENDING**, the others continue, and `outputs/run_summary.json` records the
status. The exit code is non-zero only if a task fails with an unexpected error.

### YOLO model download

The first YOLO run downloads `yolo11n.pt` (≈5.4 MB) from the official
Ultralytics GitHub release into `models/`. This needs internet once; afterwards
it runs offline. Ultralytics' settings file is kept in
`models/ultralytics_settings/` with usage analytics switched off. If the
download fails, Task 5A still runs the Haar cascade and marks YOLO as pending.

## Tests

```
python -m pytest
```

31 tests cover noise, filters and metrics, thresholding/morphology/Canny,
watershed on touching discs, HOG length, SIFT rotation/scale matching,
Haar loading and detection, YOLO (skipped if the weights are not downloaded),
video validation, frame differencing, Lucas–Kanade on a known shift, the full
motion pipeline on a generated clip, and `main.py` with missing and present
inputs. The tests never touch your personal photo or video.

## Report

```
python tools/build_report.py      # reports/CCA3_Technical_Report.docx (all numbers read from outputs/)
python tools/export_pdf.py        # optional: updates the contents page and saves a PDF via Microsoft Word
```

Fill in the yellow-highlighted fields (name, PRN, Git URL, video link) in Word.
If you rebuild the report later, those manual edits are overwritten, so fill
them in last.

## Algorithms in brief

Full beginner-friendly explanations: [`docs/ALGORITHM_EXPLANATIONS.md`](docs/ALGORITHM_EXPLANATIONS.md).

- **Gaussian filter** – weighted average of neighbours; good for small random noise, blurs edges.
- **Median filter** – median of neighbours; removes isolated black/white pixels while keeping edges.
- **PSNR / SSIM** – similarity to the clean original (higher = closer).
- **Otsu threshold** – picks the threshold that best splits the histogram into two groups.
- **Erosion / dilation / opening / closing** – shrink, grow, remove specks, fill holes.
- **Canny** – smoothing, gradients, thin edges, hysteresis thresholds.
- **Watershed** – floods from markers; distance-transform peaks separate touching objects.
- **HOG** – histograms of gradient directions in small cells (shape descriptor).
- **SIFT** – scale- and rotation-invariant keypoints with 128-value descriptors.
- **Haar cascade** – Viola–Jones boosted rectangle features for frontal faces.
- **YOLO** – one-pass neural network detector for 80 object classes.
- **Frame differencing / MOG2 / Lucas–Kanade** – changed pixels, learned background, point motion vectors.

## Results at a glance (delivered run, 28 Oct 2026)

- Salt-and-pepper noise (5 %): noisy 17.98 dB → median 3×3 **47.43 dB** (SSIM 0.995); best Gaussian filter 31.73 dB.
- Gaussian noise (σ = 25): noisy 20.44 dB → Gaussian 11×11 34.52 dB, median 11×11 34.42 dB (similar).
- Watershed on the coins sample: 22 blobs after thresholding → 24 regions (one per coin).
- SIFT: 545 keypoints; ≥ 95 % correct matches at every tested rotation, 84 % at 0.25× scale.
- Haar found the face upright but missed it at 30°, 45° and 90°, and gave false boxes with minNeighbors 1–2;
  YOLO found the person at every angle (0.91 upright) but returned a duplicate box at 90°.

Full tables and figures: `reports/CCA3_Technical_Report.pdf`.

## Common errors

| Message | Fix |
|---|---|
| `Missing input 'original_video'` | Add `inputs/original_video.mp4` (or .mov/.avi/.mkv). |
| `OpenCV cannot open the video` | Re-export the clip as H.264 MP4, or try another extension. |
| `WinError 1114 ... c10.dll` when importing torch | Venv was made from Anaconda's Python – see the setup note above. |
| `Could not load or download YOLO weights` | Check the internet connection once, or copy `yolo11n.pt` into `models/`. |
| `This OpenCV build has no SIFT` | Install `opencv-python` 4.4 or newer (see requirements.txt). |
| `No module named 'src'` | Run commands from the project folder (where `main.py` is). |
| Figures do not open with `--show` | Tkinter is missing; figures are still saved in `outputs/`. |

## Git and privacy

`.gitignore` keeps `inputs/` (personal photo/video), `.venv/`, model weights and
large output videos out of the repository. The generated figures in `outputs/`
are derived from the portrait, so make the GitHub repository **private** and add
the instructor as a collaborator. Step-by-step commands are in
[`docs/SUBMISSION_CHECKLIST.md`](docs/SUBMISSION_CHECKLIST.md).

## Limitations / pending

- Two-photo matching waits for `inputs/second_photo.jpg`.
- Motion estimation and detection on video frames wait for `inputs/original_video.mp4`.
- The watershed touching-objects experiment uses a public sample unless you add `inputs/watershed_photo.jpg`.
- The explanation video and the Git repository URL must be added to the report by the student.
