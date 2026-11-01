# Project explanation video – script and demo checklist (5–8 minutes)

The assignment requires **your own words** and **you speaking** – a silent screen
recording or slides alone are not enough. Use the points below as a guide, not
something to read word for word. Record after adding the second photo and the
video, re-running `python main.py --all` and rebuilding the report, so you can
show real motion results.

Suggested tools: OBS Studio or the Windows Snipping Tool (screen + microphone),
webcam in a corner if allowed. Upload to the instructor's Google Drive folder and
paste the link into the report (Section 11).

---

## 0:00 – 0:40  Introduction and objectives
- Your name, PRN, course; "CCA 3 practical assignment, Units 3, 4 and 5".
- Goal: apply restoration, segmentation, feature extraction, recognition and
  motion estimation to *my own* photo and video, and compare techniques with measurements.
- Tools: Python, OpenCV, NumPy, Matplotlib, scikit-image, Ultralytics YOLO.

**Show:** the project folder in VS Code / Explorer; README.

## 0:40 – 1:30  Input data
- Original photo: what it shows, how/where you took it, size (1233 × 1275).
  Why it suits the tasks: one face, plain backdrop, fine detail in hair/eyes/shirt.
- Second photo: what changed (angle, distance, lighting).
- Video: length, what moves, where the camera was, background.
- Mention that originals are never modified and personal media stays out of Git.

**Show:** `inputs/` folder; play 3–4 seconds of the video.

## 1:30 – 2:45  Unit 3 – restoration and segmentation
- Gaussian vs salt-and-pepper noise (explain in one sentence each).
- Gaussian filter = weighted average; median filter = middle value.
- Result: median 3×3 restored salt-and-pepper to about 47 dB vs about 32 dB for the
  best Gaussian; for Gaussian noise both filters were similar (~34.5 dB at 11×11).
- PSNR/SSIM higher = closer to original, but show the zoom figure: big kernels blur eyelashes.
- Otsu threshold (102), then erosion/dilation/opening/closing – read the blob/hole counts.
- Canny: noise added false edges, median filtering removed them.
- Watershed: explain the distance transform and markers; coins sample is
  supplementary (portrait has no touching objects); person/background watershed on the portrait.

**Show:** `fig_salt_pepper_comparison.png`, `fig_kernel_size_tradeoff.png`,
`fig_morphology.png`, `fig_watershed_objects.png`; open `src/image_restoration.py`
and point to `add_salt_and_pepper_noise` and `compare_with_reference`.

## 2:45 – 4:00  Unit 4 – HOG and SIFT
- HOG: gradient-direction histograms per cell; smaller cells = more detail, longer vector.
- SIFT: keypoints at many scales, 128-value descriptor relative to orientation.
- Matching: BFMatcher + ratio test 0.75 + RANSAC (green inliers, red outliers).
- Your two-photo result (number of matches / inliers) and why it is lower than for
  a rotated copy (real viewpoint and lighting change).
- Rotation: ≥95 % correct at every angle; 90°/180° keep more matches than 15–45° (interpolation).
  Resize: matches drop at 0.25× because small images lose detail.

**Show:** `fig_hog.png`, the match images, `fig_rotation_scale_sweep.png`;
`src/feature_extraction.py` → `ratio_test_matches` and `match_experiment`.

## 4:00 – 5:15  Unit 5A – Haar cascade vs YOLO
- Haar: rectangle features, integral image, boosted cascade; face only, no score.
- YOLO: one neural network, 80 classes, box + label + confidence.
- Results: both succeed upright (Haar 1 face, YOLO person 0.91).
- Failure cases you actually observed: Haar misses the face at 30°, 45°, 90°;
  false boxes with minNeighbors 1–2; YOLO duplicate box at 90°.
- On your video frames: what each detector found or missed.

**Show:** `fig_haar_vs_yolo_portrait.png`, `fig_haar_min_neighbors.png`,
`fig_rotation_robustness.png`; `src/object_detection.py` → `detect_haar`, `detect_yolo`.

## 5:15 – 6:45  Unit 5B – motion estimation
- Frame differencing: changed pixels between consecutive frames.
- MOG2: learned background model (Gaussian mixture per pixel).
- Lucas–Kanade: motion vectors for corner points.
- Read your comparison table (mean foreground %, ms per frame, moving points) and
  describe what you see: hollow masks, shadows, camera shake, warm-up of MOG2.

**Show:** `motion_preview.gif` or `motion_side_by_side.mp4`,
`fig_motion_snapshots.png`, `fig_motion_timeseries.png`.

## 6:45 – 7:30  Live demo and conclusion
- Run `python main.py --unit 3` (or `--check-inputs`) in the terminal and show
  that outputs are regenerated; mention `python -m pytest` (34 tests).
- Summarise: best technique per task, one limitation each, what you learned.

---

## Demo checklist (tick before recording)

- [ ] `inputs/second_photo.jpg` and `inputs/original_video.mp4` added
- [ ] `python main.py --all` finished without FAILED tasks
- [ ] `python tools/build_report.py` and `python tools/export_pdf.py` re-run
- [ ] Report placeholders filled (name, PRN, Git URL; video link after upload)
- [ ] Figures opened in tabs in the order above
- [ ] Terminal ready in the project folder with `.venv` activated
- [ ] Microphone tested; notifications off; personal tabs closed
- [ ] Video length between 5 and 8 minutes
- [ ] Uploaded to the instructor's Google Drive folder; link shared with the instructor
