import cv2
import numpy as np

from src import segmentation as seg


def test_otsu_separates_two_levels():
    gray = np.full((100, 100), 200, dtype=np.uint8)
    gray[30:70, 30:70] = 40
    value, mask = seg.otsu_threshold(gray, blur_kernel=3, invert=True)
    assert 40 < value < 200
    assert mask[50, 50] == 255 and mask[5, 5] == 0  # dark square becomes white


def test_opening_removes_speck_and_closing_fills_hole():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    mask[38:41, 38:41] = 0  # small hole inside the square
    mask[5, 5] = 255  # isolated white speck
    kernel = seg.structuring_element(5, "ellipse")
    results = seg.apply_morphology(mask, kernel)
    assert results["Opening"][5, 5] == 0
    assert results["Closing"][39, 39] == 255
    assert seg.mask_statistics(mask)["holes"] == 1
    assert seg.mask_statistics(results["Closing"])["holes"] == 0
    assert (results["Erosion"] > 0).sum() < (mask > 0).sum() < (results["Dilation"] > 0).sum()


def test_canny_finds_square_border():
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[25:75, 25:75] = 255
    edges = seg.canny_edges(gray, 50, 150, blur_kernel=3)
    assert edges[25:75, 20:30].any()  # left border
    assert not edges[45:55, 45:55].any()  # flat interior has no edges


def test_watershed_splits_touching_discs():
    discs = seg.synthetic_touching_discs(300)
    result = seg.watershed_objects(discs, 0.08, 0.0005, background_kernel_fraction=0)
    assert result["connected_components"] == 1  # thresholding alone sees one blob
    assert result["watershed_regions"] == 3  # watershed separates the three discs


def test_watershed_person_background_on_plain_backdrop():
    image = np.full((200, 160, 3), (190, 185, 180), dtype=np.uint8)  # plain grey backdrop
    cv2.ellipse(image, (80, 140), (50, 70), 0, 0, 360, (40, 30, 30), -1)  # dark "person"
    result = seg.watershed_person_background(image)
    mask = result["person_mask"] > 0
    assert mask[140, 80] and not mask[5, 5]
    assert 15 < result["person_pct"] < 50
