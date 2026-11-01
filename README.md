# CCA 3 – Computer Vision Practical Assignment (Units 3, 4 and 5)

**Repository:** https://github.com/Kritikakumrawat/computer-vision-cca3 (branch `main`)  
**Explanation video:** https://drive.google.com/file/d/1KC3AiaGDgUWXWRE5pkoV7XZHZk3L9u9n/view?usp=drivesdk  
**Student:** Kritika Kumrawat · PRN 1272251096 · MCA E · Instructor: Rahul Inchal · Submission: 3 November 2026

## Project overview

A Python project that applies classic and deep-learning computer-vision
techniques to the student's own photograph and self-recorded videos, measures
the results and compares the techniques. One command (`python main.py --all`)
runs everything and writes every image, figure, table and JSON result to
`outputs/`; a second command builds the technical report (DOCX + PDF) from
those results, so every number in the report comes from an actual run.

## Technologies

| Technology | Used for |
|---|---|
| Python 3.11 | Language (tested on Windows 11, CPU only) |
| OpenCV 4.14 | Filtering, thresholding, morphology, Canny, watershed, SIFT, matching, RANSAC, Haar cascades, MOG2, Lucas–Kanade, video I/O |
| NumPy | Image arrays, seeded synthetic noise, statistics |
| scikit-image | PSNR/SSIM, HOG, peak detection and watershed, sample image |
| SciPy | Hole filling, labelling |
| Matplotlib | All comparison figures and charts |
| Ultralytics YOLO11n (PyTorch CPU) | Pretrained object detector (COCO, 80 classes) |
| Pillow | GIF output |
| python-docx, pywin32 (Word) | Report generation and PDF export |
| pytest | Automated tests |

## Features

- Every assignment task in its own module, run through one entry point with `--unit 3/4/5` or `--all`.
- All parameters in `src/config.py`; fixed random seed, so results are repeatable.
- Missing inputs never crash a run: tasks are marked **PENDING** and the rest continue.
- Measured comparisons: PSNR/SSIM for every filter and kernel size, blob/hole counts, edge densities, SIFT correct-match rates under rotation/scale/lighting, Haar vs YOLO robustness tests, per-frame motion statistics and timings.
- Genuine execution screenshots captured automatically from real console and figure windows.
- Report built automatically from the outputs (DOCX + PDF), 34 automated tests.

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
| `inputs/original_video.mp4` | Present (24.7 s self-recorded clip, 832×464, 60 fps) | Unit 5B motion estimation and Unit 5A video-frame detection complete |
| `inputs/front_camera_video.mp4` | Present (5.9 min self-recorded front-camera clip) | Optional: 5 frames used for Haar/YOLO face detection in Unit 5A |
| `inputs/second_photo.png` | Present (463×609 close-up photo of the student) | Unit 4 two-photo matching, plus rotation/resizing of the second photo |
| `inputs/second_photo_b.png` | Present (277×399 wider photo, same scene as the second photo) | Optional extra same-scene matching pair in Unit 4 |

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
│   ├── motion_estimation.py # Unit 5B
│   └── screen_capture.py    # Windows window capture for the execution screenshots
├── inputs/                  # your photo / video (kept out of Git)
├── outputs/
│   ├── unit3/restoration/   # noisy + restored images, metrics CSV, figures
│   ├── unit3/segmentation/  # masks, morphology, edges, watershed
│   ├── unit4/               # HOG, SIFT keypoints, matches, sweep tables
│   ├── unit5/detection/     # Haar / YOLO images and tables
│   ├── unit5/motion/        # frames, masks, motion vectors, GIF (side-by-side MP4 is local only)
│   └── run_summary.json     # status of every task in the last run
├── reports/                 # CCA3_Technical_Report.docx / .pdf
├── docs/                    # algorithm explanations, video script, submission checklist
├── tests/                   # pytest suite (synthetic + public sample data only)
├── tools/                   # capture_screenshots.py, extract_video_frame.py, build_report.py, export_pdf.py
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
python main.py --unit 3 --screenshots   # Windows: show the main figures and save a screenshot of each window
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

34 tests cover noise, filters and metrics, thresholding/morphology/Canny,
watershed on touching discs, HOG length, SIFT rotation/scale matching,
Haar loading and detection, YOLO (skipped if the weights are not downloaded),
video validation, frame differencing, Lucas–Kanade on a known shift, the full
motion pipeline on a generated clip, and `main.py` with missing and present
inputs, plus the screenshot option (without opening windows). The tests never touch
your personal photo or video.

## Execution screenshots

```
python tools/capture_screenshots.py              # all steps (about 4 minutes)
python tools/capture_screenshots.py unit5        # only one step, e.g. after adding the video
```

For each step (input check, Unit 3, Unit 4, Unit 5, pytest) the tool opens a
real console window, runs the command there and, when it has finished, captures
that window with the Windows `PrintWindow` function. The unit runs use
`main.py --screenshots`, so the result figures appear in live windows during the
run and each window is captured too. Only the program's own windows are
captured. Windows pop up for a few minutes – do not close them by hand.

Where the screenshots are saved (no extra folder – next to the results they show):

| Screenshot | Location |
|---|---|
| Console runs | `outputs/screenshot_terminal_check_inputs.png`, `outputs/screenshot_terminal_pytest.png`, `outputs/unit3/screenshot_terminal_unit3.png`, `outputs/unit4/screenshot_terminal_unit4.png`, `outputs/unit5/screenshot_terminal_unit5.png` |
| Result windows | `screenshot_<figure>.jpg` beside each figure, e.g. `outputs/unit3/restoration/screenshot_fig_salt_pepper_comparison.jpg` |
| Index (used by the report) | `outputs/screenshots_index.json` |

Requires Windows and `pywin32` (in requirements.txt).

## Report

```
python tools/build_report.py      # reports/CCA3_Technical_Report.docx (all numbers read from outputs/)
python tools/export_pdf.py        # optional: updates the contents page and saves a PDF via Microsoft Word
```

The report includes the screenshots in Section 8 ("Execution screenshots").
Student details and the repository URL are set at the top of
`tools/build_report.py`, together with the explanation-video link. Rebuilding
overwrites any manual edits made in Word.

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

## Results at a glance (run of 1 Nov 2026)

- Salt-and-pepper noise (5 %): noisy 17.98 dB → median 3×3 **47.43 dB** (SSIM 0.995); best Gaussian filter 31.73 dB.
- Gaussian noise (σ = 25): noisy 20.44 dB → Gaussian 11×11 34.52 dB, median 11×11 34.42 dB (similar).
- Watershed on the coins sample: 22 blobs after thresholding → 24 regions (one per coin).
- SIFT: 545 keypoints; ≥ 95 % correct matches at every tested rotation, 84 % at 0.25× scale.
  Portrait vs second photo: only 19 ratio-test matches and none connects the same feature (different place,
  clothes, hair and lighting); the two small same-scene photos gave 14 matches, none confirmed – SIFT is
  rotation/scale invariant but cannot match different scenes, and low resolution limits it further.
- Haar found the face upright but missed it at 30°, 45° and 90°, and gave false boxes with minNeighbors 1–2;
  YOLO found the person at every angle (0.91 upright) but returned a duplicate box at 90°.
- Video (24.7 s, 1479 frames): frame differencing marked 2.07 % of each frame on average, MOG2 4.33 %;
  Lucas–Kanade tracked about 200 corners, about 23 of them moving at 2.07 px/frame. YOLO found the person
  and the bed in the sampled frames; the frontal-face Haar cascade found no face (back and side views).
- Front-camera frames: Haar found the face in all 5 frames but added a false face box in 2 and extra eye boxes;
  YOLO found the person in every frame (0.92–0.95).

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

Repository: https://github.com/Kritikakumrawat/computer-vision-cca3 (branch `main`).

`.gitignore` keeps `inputs/` (personal photo and videos), `.venv/`, `.tools/`,
model weights, large output videos, caches, environment/secret files and
temporary files out of the repository. The original media are submitted
through Google Drive (and inside the submission ZIP) instead, as the assignment
asks for large raw videos. The generated figures in `outputs/` and the report
show the student's face; keep that in mind if the repository is public. Upload
commands are in [`docs/SUBMISSION_CHECKLIST.md`](docs/SUBMISSION_CHECKLIST.md).

## Limitations / pending

- The second photographs are small PNG copies (277×399 and 463×609, no camera metadata); the phone's full-resolution originals would give SIFT more detail to match.
- The video was received through WhatsApp, which re-encodes it; if the camera's original file is available, replace `inputs/original_video.mp4` and re-run.
- The watershed touching-objects experiment uses a public sample unless you add `inputs/watershed_photo.jpg`.
- `tools/extract_video_frame.py` can extract a labelled face frame from a video as a Unit 4 fallback; it is not used now that a second photograph exists.
