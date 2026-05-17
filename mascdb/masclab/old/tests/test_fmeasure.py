"""
Unit tests for fmeasure module.

Tests the fmeasure function to ensure it behaves like the MATLAB version.
"""

import pytest
import numpy as np
from src.features.fmeasure import fmeasure


class TestFmeasureLAPM:
    """Test Modified Laplacian (LAPM) focus measure."""
    
    def test_lapm_uniform_image(self):
        """Test LAPM on uniform image (should be near zero)."""
        image = np.ones((50, 50), dtype=np.uint8) * 128
        fm = fmeasure(image, 'LAPM', None)
        
        # Uniform image should have very low LAPM (near zero)
        assert fm < 1.0
        assert fm >= 0.0
    
    def test_lapm_gradient_image(self):
        """Test LAPM on gradient image (should be positive)."""
        # Create horizontal gradient
        image = np.tile(np.arange(0, 256, dtype=np.uint8), (50, 1))
        fm = fmeasure(image, 'LAPM', None)
        
        # Gradient should have positive LAPM
        assert fm > 0.0
    
    def test_lapm_sharp_edges(self):
        """Test LAPM on image with sharp edges (should be high)."""
        image = np.zeros((50, 50), dtype=np.uint8)
        # Add sharp vertical and horizontal edges
        image[20:30, :] = 255
        image[:, 20:30] = 255
        
        fm = fmeasure(image, 'LAPM', None)
        
        # Sharp edges should have measurable LAPM
        assert fm > 10.0  # Realistic threshold for this pattern
    
    def test_lapm_random_texture(self):
        """Test LAPM on random texture."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        fm = fmeasure(image, 'LAPM', None)
        
        # Random texture should have positive LAPM
        assert fm > 0.0
        assert not np.isnan(fm)
        assert not np.isinf(fm)
    
    def test_lapm_with_roi(self):
        """Test LAPM with ROI parameter."""
        # Create image with feature in top-left corner
        image = np.zeros((100, 100), dtype=np.uint8)
        image[10:30, 10:30] = 255
        
        # Measure full image
        fm_full = fmeasure(image, 'LAPM', None)
        
        # Measure only ROI containing the feature
        fm_roi = fmeasure(image, 'LAPM', (5, 5, 30, 30))
        
        # ROI should have higher focus than full image
        assert fm_roi > fm_full
    
    def test_lapm_case_insensitive(self):
        """Test that measure name is case-insensitive."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        fm_upper = fmeasure(image, 'LAPM', None)
        fm_lower = fmeasure(image, 'lapm', None)
        fm_mixed = fmeasure(image, 'LaPm', None)
        
        assert fm_upper == fm_lower == fm_mixed


class TestFmeasureHISE:
    """Test Histogram Entropy (HISE) focus measure."""
    
    def test_hise_uniform_image(self):
        """Test HISE on uniform image (should be zero)."""
        image = np.ones((50, 50), dtype=np.uint8) * 128
        fm = fmeasure(image, 'HISE', None)
        
        # Uniform image should have zero entropy
        assert fm == 0.0
    
    def test_hise_two_levels(self):
        """Test HISE on image with two intensity levels."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[:25, :] = 0
        image[25:, :] = 255
        
        fm = fmeasure(image, 'HISE', None)
        
        # Two equal levels should give entropy = 1
        assert 0.9 < fm < 1.1
    
    def test_hise_random_image(self):
        """Test HISE on random image (should be high)."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        fm = fmeasure(image, 'HISE', None)
        
        # Random image should have high entropy
        assert fm > 4.0  # Typical range for random 8-bit image
        assert fm < 8.0  # Max entropy for 256 bins
        assert not np.isnan(fm)
    
    def test_hise_gradient(self):
        """Test HISE on gradient image."""
        # Create gradient with all intensity values
        image = np.zeros((50, 256), dtype=np.uint8)
        for i in range(256):
            image[:, i] = i
        
        fm = fmeasure(image, 'HISE', None)
        
        # Should have high entropy (many different levels)
        assert fm > 5.0
    
    def test_hise_with_roi(self):
        """Test HISE with ROI parameter."""
        # Create image with different regions
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 100  # Uniform top half
        np.random.seed(42)
        image[50:, :] = np.random.randint(0, 255, (50, 100), dtype=np.uint8)  # Random bottom half
        
        # Measure ROI on uniform part
        fm_uniform = fmeasure(image, 'HISE', (0, 0, 100, 50))
        
        # Measure ROI on random part
        fm_random = fmeasure(image, 'HISE', (0, 50, 100, 50))
        
        # Random region should have higher entropy
        assert fm_random > fm_uniform


class TestFmeasureWAVS:
    """Test Wavelet Sum (WAVS) focus measure."""
    
    def test_wavs_uniform_image(self):
        """Test WAVS on uniform image (should be near zero)."""
        image = np.ones((50, 50), dtype=np.uint8) * 128
        fm = fmeasure(image, 'WAVS', None)
        
        # Uniform image should have low wavelet coefficients
        assert fm < 10.0
    
    def test_wavs_edges(self):
        """Test WAVS on image with edges."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[20:30, :] = 255
        
        fm = fmeasure(image, 'WAVS', None)
        
        # Edges should produce significant wavelet coefficients
        assert fm > 0.0
        assert not np.isnan(fm)
    
    def test_wavs_random_texture(self):
        """Test WAVS on random texture."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        fm = fmeasure(image, 'WAVS', None)
        
        # Random texture should have positive WAVS
        assert fm > 0.0
        assert not np.isnan(fm)
        assert not np.isinf(fm)
    
    def test_wavs_with_roi(self):
        """Test WAVS with ROI parameter."""
        # Create image with feature in corner
        image = np.zeros((100, 100), dtype=np.uint8)
        np.random.seed(42)
        image[10:40, 10:40] = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
        
        # Measure full image
        fm_full = fmeasure(image, 'WAVS', None)
        
        # Measure only ROI with feature
        fm_roi = fmeasure(image, 'WAVS', (5, 5, 40, 40))
        
        # ROI with feature should have higher WAVS
        assert fm_roi > fm_full


class TestFmeasureComparison:
    """Test comparison between different focus measures."""
    
    def test_all_basic_measures_on_same_image(self):
        """Test that basic measures (used in process_basic_descriptors) work."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        lapm = fmeasure(image, 'LAPM', None)
        hise = fmeasure(image, 'HISE', None)
        wavs = fmeasure(image, 'WAVS', None)
        
        # All should return valid numbers
        assert lapm > 0
        assert hise > 0
        assert wavs > 0
        
        assert not np.isnan(lapm)
        assert not np.isnan(hise)
        assert not np.isnan(wavs)
    
    def test_additional_measures(self):
        """Test additional focus measures for completeness."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        # Test a selection of additional measures
        measures_to_test = ['GLVA', 'GRAE', 'TENG', 'LAPE', 'LAPV', 'BREN', 'GRAS', 'SFRQ']
        
        for measure in measures_to_test:
            fm = fmeasure(image, measure, None)
            assert not np.isnan(fm), f"{measure} returned NaN"
            assert not np.isinf(fm), f"{measure} returned Inf"
    
    def test_all_measures_no_crash(self):
        """Test that all 28 measures run without crashing."""
        np.random.seed(42)
        image = np.random.randint(50, 200, (50, 50), dtype=np.uint8)
        
        all_measures = [
            'ACMO', 'BREN', 'CONT', 'CURV', 'DCTE', 'DCTR',
            'GDER', 'GLVA', 'GLLV', 'GLVN', 'GRAE', 'GRAT',
            'GRAS', 'HELM', 'HISE', 'HISR', 'LAPE', 'LAPM',
            'LAPV', 'LAPD', 'SFIL', 'SFRQ', 'TENG', 'TENV',
            'VOLA', 'WAVS', 'WAVV', 'WAVR'
        ]
        
        results = {}
        for measure in all_measures:
            try:
                fm = fmeasure(image, measure, None)
                results[measure] = fm
                assert not np.isnan(fm), f"{measure} returned NaN"
            except Exception as e:
                pytest.fail(f"Measure {measure} raised exception: {e}")
        
        # Verify we got results for all measures
        assert len(results) == 28
    
    def test_focused_vs_blurred(self):
        """Test that focused image scores higher than blurred."""
        # Create sharp checkerboard pattern
        image_sharp = np.zeros((50, 50), dtype=np.uint8)
        for i in range(0, 50, 10):
            for j in range(0, 50, 10):
                if (i + j) // 10 % 2 == 0:
                    image_sharp[i:i+10, j:j+10] = 255
        
        # Create blurred version (simple averaging)
        from scipy import ndimage
        image_blur = ndimage.uniform_filter(image_sharp.astype(float), size=5).astype(np.uint8)
        
        # Measure both
        lapm_sharp = fmeasure(image_sharp, 'LAPM', None)
        lapm_blur = fmeasure(image_blur, 'LAPM', None)
        
        # Sharp image should have higher LAPM
        assert lapm_sharp > lapm_blur


class TestFmeasureEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_measure(self):
        """Test that invalid measure raises error."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Unknown measure"):
            fmeasure(image, 'INVALID', None)
    
    def test_empty_roi(self):
        """Test with ROI that results in empty image."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        # ROI with zero width or height
        fm = fmeasure(image, 'LAPM', (10, 10, 0, 0))
        
        # Should handle gracefully (returns 0.0 for empty ROI)
        assert fm == 0.0 or not np.isnan(fm)
    
    def test_small_image(self):
        """Test with very small image."""
        image = np.random.randint(0, 255, (5, 5), dtype=np.uint8)
        
        lapm = fmeasure(image, 'LAPM', None)
        hise = fmeasure(image, 'HISE', None)
        wavs = fmeasure(image, 'WAVS', None)
        
        # Should not crash
        assert not np.isnan(lapm)
        assert not np.isnan(hise)
        assert not np.isnan(wavs)
    
    def test_float_image(self):
        """Test with float image instead of uint8."""
        image = np.random.rand(50, 50) * 255
        
        fm = fmeasure(image, 'LAPM', None)
        
        # Should handle float images
        assert fm > 0
        assert not np.isnan(fm)
    
    def test_roi_out_of_bounds(self):
        """Test with ROI partially out of bounds."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        # ROI extends beyond image
        # Note: This will just crop to valid region
        fm = fmeasure(image, 'LAPM', (40, 40, 20, 20))
        
        # Should handle gracefully (crop to 10x10)
        assert not np.isnan(fm)


class TestFmeasureReproducibility:
    """Test that results are reproducible."""
    
    def test_same_image_same_result(self):
        """Test that same image gives same result."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        fm1 = fmeasure(image, 'LAPM', None)
        fm2 = fmeasure(image, 'LAPM', None)
        
        assert fm1 == fm2
    
    def test_different_seeds_different_results(self):
        """Test that different random images give different results."""
        np.random.seed(42)
        image1 = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        np.random.seed(123)
        image2 = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        fm1 = fmeasure(image1, 'LAPM', None)
        fm2 = fmeasure(image2, 'LAPM', None)
        
        # Different images should likely give different results
        # (not guaranteed but very likely with random data)
        assert abs(fm1 - fm2) > 0.01


class TestFmeasureGradientBased:
    """Test gradient-based focus measures."""
    
    def test_grae_gradient(self):
        """Test GRAE (Energy of gradient)."""
        # Create horizontal gradient
        image = np.tile(np.arange(0, 256, dtype=np.uint8), (50, 1))
        fm = fmeasure(image, 'GRAE', None)
        
        assert fm > 0
        assert not np.isnan(fm)
    
    def test_teng_edges(self):
        """Test TENG (Tenengrad)."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[20:30, :] = 255
        
        fm = fmeasure(image, 'TENG', None)
        
        assert fm > 0
        assert not np.isnan(fm)
    
    def test_gras_horizontal_gradient(self):
        """Test GRAS (Squared gradient)."""
        image = np.tile(np.arange(0, 256, dtype=np.uint8), (50, 1))
        fm = fmeasure(image, 'GRAS', None)
        
        assert fm > 0
        assert not np.isnan(fm)


class TestFmeasureLaplacianVariants:
    """Test different Laplacian-based measures."""
    
    def test_lape_vs_lapm_vs_lapv(self):
        """Test that different Laplacian measures give different results."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        lape = fmeasure(image, 'LAPE', None)
        lapm = fmeasure(image, 'LAPM', None)
        lapv = fmeasure(image, 'LAPV', None)
        
        # All should be valid
        assert lape > 0 and not np.isnan(lape)
        assert lapm > 0 and not np.isnan(lapm)
        assert lapv >= 0 and not np.isnan(lapv)
        
        # They should give different values
        assert lape != lapm or lapm != lapv
    
    def test_lapd_diagonal(self):
        """Test LAPD (Diagonal Laplacian)."""
        # Create diagonal pattern
        image = np.zeros((50, 50), dtype=np.uint8)
        for i in range(50):
            image[i, i] = 255
        
        fm = fmeasure(image, 'LAPD', None)
        
        assert fm > 0
        assert not np.isnan(fm)


class TestFmeasureStatistical:
    """Test statistical focus measures."""
    
    def test_glva_variance(self):
        """Test GLVA (Graylevel variance)."""
        # Uniform image should have low variance
        uniform = np.ones((50, 50), dtype=np.uint8) * 128
        fm_uniform = fmeasure(uniform, 'GLVA', None)
        
        # Random image should have higher variance
        np.random.seed(42)
        random = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        fm_random = fmeasure(random, 'GLVA', None)
        
        assert fm_random > fm_uniform
    
    def test_hisr_range(self):
        """Test HISR (Histogram range)."""
        # Image with small range
        small_range = np.ones((50, 50), dtype=np.uint8) * 100
        small_range[25:, :] = 110
        fm_small = fmeasure(small_range, 'HISR', None)
        
        # Image with large range
        large_range = np.zeros((50, 50), dtype=np.uint8)
        large_range[25:, :] = 255
        fm_large = fmeasure(large_range, 'HISR', None)
        
        assert fm_large > fm_small
        assert fm_small == 10.0
        assert fm_large == 255.0


class TestFmeasureWaveletVariants:
    """Test different wavelet-based measures."""
    
    def test_wavs_vs_wavv(self):
        """Test WAVS vs WAVV."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        wavs = fmeasure(image, 'WAVS', None)
        wavv = fmeasure(image, 'WAVV', None)
        
        # Both should be valid
        assert wavs > 0 and not np.isnan(wavs)
        assert wavv >= 0 and not np.isnan(wavv)
    
    def test_wavr_ratio(self):
        """Test WAVR (Wavelet ratio)."""
        np.random.seed(42)
        image = np.random.randint(50, 200, (50, 50), dtype=np.uint8)
        
        fm = fmeasure(image, 'WAVR', None)
        
        assert fm >= 0
        assert not np.isnan(fm)


class TestFmeasureSpecialCases:
    """Test special/complex focus measures."""
    
    def test_vola_correlation(self):
        """Test VOLA (Vollath's correlation)."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        fm = fmeasure(image, 'VOLA', None)
        
        # VOLA can be negative
        assert not np.isnan(fm)
        assert not np.isinf(fm)
    
    def test_acmo_central_moment(self):
        """Test ACMO (Absolute Central Moment)."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        fm = fmeasure(image, 'ACMO', None)
        
        assert fm >= 0
        assert not np.isnan(fm)
    
    def test_cont_contrast(self):
        """Test CONT (Image contrast)."""
        # Uniform image should have low contrast
        uniform = np.ones((50, 50), dtype=np.uint8) * 128
        fm_uniform = fmeasure(uniform, 'CONT', None)
        
        # Checkerboard should have high contrast
        checker = np.zeros((50, 50), dtype=np.uint8)
        checker[::2, ::2] = 255
        checker[1::2, 1::2] = 255
        fm_checker = fmeasure(checker, 'CONT', None)
        
        assert fm_checker > fm_uniform


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
