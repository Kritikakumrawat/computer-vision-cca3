import numpy as np
import pytest

from src import image_restoration as ir


def test_gaussian_noise_is_reproducible_and_keeps_shape(smooth_image):
    a = ir.add_gaussian_noise(smooth_image, sigma=20, seed=1)
    b = ir.add_gaussian_noise(smooth_image, sigma=20, seed=1)
    c = ir.add_gaussian_noise(smooth_image, sigma=20, seed=2)
    assert a.shape == smooth_image.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)  # same seed -> same noise
    assert not np.array_equal(a, c)


def test_salt_and_pepper_amount(smooth_image):
    noisy = ir.add_salt_and_pepper_noise(smooth_image, amount=0.1, seed=3)
    changed = np.any(noisy != smooth_image, axis=2).mean()
    assert 0.05 < changed <= 0.11  # about 10 % (some pixels were already black/white)
    corrupted = noisy[np.any(noisy != smooth_image, axis=2)]
    assert set(np.unique(corrupted)) <= {0, 255}


def test_identical_images_give_perfect_metrics(smooth_image):
    metrics = ir.compare_with_reference(smooth_image, smooth_image.copy())
    assert metrics["psnr_db"] == float("inf")
    assert metrics["ssim"] == 1.0


def test_shape_mismatch_is_rejected(smooth_image):
    with pytest.raises(ValueError):
        ir.compare_with_reference(smooth_image, smooth_image[:-1])


def test_median_beats_gaussian_on_salt_and_pepper(smooth_image):
    noisy = ir.add_salt_and_pepper_noise(smooth_image, amount=0.05, seed=4)
    median = ir.compare_with_reference(smooth_image, ir.median_filter(noisy, 3))
    gaussian = ir.compare_with_reference(smooth_image, ir.gaussian_filter(noisy, 3))
    noisy_metrics = ir.compare_with_reference(smooth_image, noisy)
    assert median["psnr_db"] > gaussian["psnr_db"] > noisy_metrics["psnr_db"]
    assert median["ssim"] > noisy_metrics["ssim"]


def test_filters_improve_gaussian_noise(smooth_image):
    noisy = ir.add_gaussian_noise(smooth_image, sigma=25, seed=5)
    before = ir.compare_with_reference(smooth_image, noisy)["psnr_db"]
    for fn in (ir.gaussian_filter, ir.median_filter):
        assert ir.compare_with_reference(smooth_image, fn(noisy, 5))["psnr_db"] > before
