# Algorithm explanations (beginner-friendly)

Each section says **what the algorithm does**, **how it works in a few steps**,
**where it is in the code**, and **what to watch out for**. An image is just a
grid of numbers: grayscale pixels are 0 (black) to 255 (white); colour pixels
have three such numbers (OpenCV stores them as Blue, Green, Red).

---

## Unit 3A – Noise, filters and quality metrics

### Gaussian noise
- **What:** small random changes added to *every* pixel, like sensor grain in low light.
- **How:** for each pixel and channel, draw a random number from a normal
  distribution with mean 0 and standard deviation σ (here 25) and add it; clip to 0–255.
- **Code:** `add_gaussian_noise()` in `src/image_restoration.py`. A fixed seed (42)
  makes the noise identical on every run.

### Salt-and-pepper (impulse) noise
- **What:** a few pixels become pure white (salt) or pure black (pepper), like
  dead pixels or transmission errors.
- **How:** pick 5 % of pixel positions at random; set half to 255 and half to 0.
- **Code:** `add_salt_and_pepper_noise()`.

### Gaussian filter (blur)
- **What:** replaces each pixel by a weighted average of its neighbours; the
  centre counts most, far pixels count less (bell-shaped weights).
- **Good for:** Gaussian noise. **Weakness:** blurs edges; spreads impulse noise
  into grey spots instead of removing it.
- **Code:** `cv2.GaussianBlur(image, (k, k), 0)` – σ is computed from the kernel size k.

### Median filter
- **What:** replaces each pixel by the *median* (middle value after sorting) of its k×k neighbourhood.
- **Why it beats impulse noise:** a single 0 or 255 is an outlier; it ends up at
  one end of the sorted list and is never chosen. Edges stay sharp because the
  output is always a real neighbouring value.
- **Code:** `cv2.medianBlur(image, k)`. OpenCV is very fast for k ≤ 5 and slower above.

### PSNR (Peak Signal-to-Noise Ratio)
- **Formula:** `PSNR = 10 · log10(255² / MSE)`, where MSE is the mean squared
  difference between the test image and the clean original.
- **Reading it:** in decibels; higher is better. ~20 dB = very noisy, 30–40 dB =
  good, > 40 dB = almost identical. Identical images give infinity.

### SSIM (Structural Similarity)
- **What:** compares local brightness, contrast and *structure* (patterns) in
  small windows; 1.0 = identical, 0 = unrelated. Closer to human perception than PSNR.
- **Code:** `skimage.metrics.structural_similarity(..., channel_axis=2, data_range=255)`.
- **Caution:** both metrics summarise the whole image. A big blur can score well
  on a photo with large smooth areas while fine detail is lost – always look at
  the zoomed figures too.

---

## Unit 3B – Thresholding, morphology, edges, watershed

### Grayscale conversion
`gray = 0.299·R + 0.587·G + 0.114·B` (`cv2.cvtColor(..., COLOR_BGR2GRAY)`).
Most segmentation steps work on one channel.

### Otsu thresholding
- **What:** turns grayscale into black/white with one threshold T.
- **How:** tries every T from 0 to 255 and picks the one that makes the two
  groups (below / above T) as compact as possible (minimum within-class variance).
- **Code:** `cv2.threshold(blurred, 0, 255, THRESH_BINARY_INV + THRESH_OTSU)`. With
  `_INV`, the darker group (hair, blazer) becomes white.

### Adaptive thresholding
Each pixel gets its own threshold = weighted mean of its 31×31 neighbourhood
minus C. Works under uneven lighting and highlights local texture/edges.

### Erosion and dilation
- A small shape called the **structuring element** (here a 9×9 ellipse) slides over the mask.
- **Erosion:** a pixel stays white only if *all* pixels under the element are white → objects shrink, thin parts vanish.
- **Dilation:** a pixel becomes white if *any* pixel under the element is white → objects grow, gaps close.

### Opening and closing
- **Opening = erosion then dilation:** removes small white specks but keeps big shapes the same size.
- **Closing = dilation then erosion:** fills small black holes and cracks inside shapes.
- **Code:** `cv2.morphologyEx(mask, MORPH_OPEN / MORPH_CLOSE, kernel)`.
- The project counts white blobs and enclosed holes before/after each operation (`mask_statistics()`).

### Top-hat (used before watershed on the coins)
`image − opening(image)` with a large element keeps only small bright objects
and removes slowly varying background light. It fixes uneven illumination.

### Canny edge detection
1. Gaussian smoothing (reduces noise).
2. Sobel gradients → edge strength and direction.
3. Non-maximum suppression → keeps only the thin ridge of each edge.
4. Hysteresis: pixels above the *high* threshold are edges; pixels between low
   and high are kept only if connected to a strong edge.
- **Code:** `cv2.Canny(blurred, 50, 150)`. Noise creates many false edges, so restore first.

### Watershed segmentation
- **Idea:** think of the image as a landscape. Water rises from chosen *markers*;
  where water from two markers meets, a dam (boundary) is built.
- **Touching objects (distance-transform watershed):**
  1. Threshold → objects white.
  2. **Distance transform:** each white pixel gets its distance to the nearest
     black pixel, so object centres are peaks.
  3. Each peak = one marker (`peak_local_max`).
  4. `skimage.segmentation.watershed(-distance, markers, mask=...)` floods from
     the markers; touching objects meet along the "neck" between them.
- **Marker-based watershed on the portrait:** markers are "sure background"
  (colour close to the backdrop, connected to the border) and "sure person"
  (colour far from the backdrop); `cv2.watershed` decides the uncertain band
  along the strongest image edges.
- **Watch out:** too many peaks → one object split into pieces (over-segmentation);
  too few → touching objects stay merged.

---

## Unit 4 – Features and matching

### HOG (Histogram of Oriented Gradients)
1. Compute gradient direction and strength for every pixel.
2. Split the image into cells (e.g. 16×16 px); in each cell build a 9-bin
   histogram of directions (0–180°), weighted by strength.
3. Group 2×2 cells into blocks and normalise each block (L2-Hys) → robust to lighting.
4. Concatenate all block histograms → one long feature vector.
- **Length** = blocks across × blocks down × 4 cells × 9 bins (checked in `hog_statistics.csv`).
- **Use:** describes *shape*; classic for pedestrian detection with an SVM.

### SIFT (Scale-Invariant Feature Transform)
1. **Scale space:** blur the image more and more and subtract neighbours
   (Difference of Gaussians); keypoints are extrema across position *and* scale.
2. Remove low-contrast points and points on edges.
3. **Orientation:** each keypoint gets its dominant gradient direction.
4. **Descriptor:** a 16×16 patch around the point, rotated to that orientation,
   split into 4×4 sub-regions with 8-bin histograms → **128 numbers**.
- Because size and angle are measured per keypoint, SIFT is invariant to scale and rotation.
- **Code:** `cv2.SIFT_create().detectAndCompute(gray, None)`.

### Matching
- **Brute force (BFMatcher):** compare every descriptor with every other using
  Euclidean (L2) distance. **FLANN:** fast approximate nearest-neighbour search (KD-trees).
- **Lowe's ratio test:** keep a match only if the best distance < 0.75 × second-best;
  ambiguous matches (repeated textures) are dropped.
- **RANSAC:** repeatedly fit a transform to a random minimal set of matches and keep
  the model most matches agree with; disagreeing matches are outliers (red lines).
  Homography for a rotated/resized copy; fundamental matrix for two real photos of a 3-D scene.
- **Correct-match rate:** for a rotated/resized copy we know exactly where each
  point should go, so we can check each match within 3 px.

### Why rotation, scale, lighting and viewpoint matter
- **Rotation:** handled by the keypoint orientation; non-90° angles need pixel interpolation, which slightly changes gradients.
- **Scale:** handled by the scale space, but shrinking an image destroys fine detail, so fewer keypoints exist.
- **Lighting:** descriptors are normalised, so uniform brightness changes matter little; strong non-linear changes reduce contrast in some areas.
- **Viewpoint:** a 3-D change of camera position distorts local patches (perspective), which SIFT only partly tolerates.

---

## Unit 5A – Object recognition

### Viola–Jones Haar cascade
1. **Haar-like features:** differences between sums of adjacent light/dark rectangles
   (e.g. eyes darker than cheeks).
2. **Integral image:** any rectangle sum in 4 look-ups → very fast.
3. **AdaBoost:** picks the few most useful features out of thousands.
4. **Cascade:** a chain of stages; most windows are rejected by the first cheap stages.
5. Window slides over an image pyramid (`scaleFactor` 1.1 = 10 % size steps);
   overlapping hits are grouped and `minNeighbors` hits are needed to accept a face.
- **Limits:** one object type per cascade, upright frontal faces only, no confidence score.

### YOLO (You Only Look Once) – Ultralytics YOLO11n
- A convolutional neural network looks at the whole image once and predicts boxes,
  class labels and confidence scores on several feature-map scales; overlapping
  boxes are merged by non-maximum suppression.
- Pretrained on COCO (80 classes such as person, car, chair, phone) – it has a
  "person" class but no "face" class.
- **Code:** `YOLO("models/yolo11n.pt").predict(image, conf=0.25)`.

---

## Unit 5B – Motion estimation

### Frame differencing
`|frame(t) − frame(t−1)|` → threshold → small cleanup. Very fast; marks only the
pixels that *changed*, so a uniformly coloured object gives a hollow outline and
an object that stops moving disappears. Any camera movement marks the whole frame.

### GMM background subtraction (MOG2)
Each pixel's colour history is modelled by a mixture of a few Gaussians.
Components that are frequent and stable = background. A new pixel value that does
not fit them = foreground. It also flags shadows (grey value 127 – removed here).
Needs some frames to learn; an object that stays still long enough becomes background.
**Code:** `cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)`.

### Lucas–Kanade optical flow
- Assumes a small patch keeps its brightness and moves by (dx, dy) between frames;
  solves for (dx, dy) by least squares in a 21×21 window.
- Pyramids (3 levels) let it follow larger motions.
- Tracks **corners** (Shi–Tomasi `goodFeaturesToTrack`) because flat areas have no
  unique motion (the *aperture problem*).
- Output: motion vectors (arrows) with direction and speed – the only one of the three methods that gives direction.

### Comparison in one line each
- **Frame differencing:** what changed since the last frame.
- **MOG2:** what is not part of the learned background.
- **Lucas–Kanade:** where each tracked point moved, and how fast.
