"""
Forensic Analyzer
=================
Low-level forensic signal extraction from face crops.
Computes texture inconsistencies, noise patterns, frequency artifacts,
and compression anomalies — key indicators of GAN-generated faces.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ForensicAnalyzer:
    """
    Extracts forensic signals from face crops for deepfake detection.
    
    Forensic signals analyzed:
    1. Laplacian sharpness / blur variance (GANs often produce over-smooth textures)
    2. High-frequency noise pattern (GANs introduce characteristic HF noise)
    3. JPEG compression artifacts (mismatched compression across face boundaries)
    4. Color channel statistics (GAN hue/saturation drift)
    5. Edge density and consistency (GAN blending boundary artifacts)
    6. Noise floor estimation via SRM (Steganalysis Rich Model-inspired)
    """

    def __init__(self):
        # SRM (noise residual) filter kernels
        self._srm_filters = self._build_srm_filters()

    def _build_srm_filters(self):
        """Build simplified SRM (Steganalysis Rich Model) noise residual filters."""
        f1 = np.array([[0, 0, 0, 0, 0],
                       [0, -1, 2, -1, 0],
                       [0, 2, -4, 2, 0],
                       [0, -1, 2, -1, 0],
                       [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0

        f2 = np.array([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
        return [f1, f2]

    def compute_laplacian_sharpness(self, gray):
        """Laplacian variance — low values indicate GAN over-smoothing."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def compute_noise_residual(self, gray):
        """Noise residual via SRM filters. GANs leave characteristic patterns."""
        residuals = []
        for kernel in self._srm_filters:
            res = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            residuals.append(float(np.std(res)))
        return float(np.mean(residuals))

    def compute_frequency_artifacts(self, gray):
        """
        Detect GAN frequency artifacts using 2D FFT.
        GANs often produce checkerboard artifacts at specific frequencies.
        Returns ratio of high-frequency energy to total energy.
        """
        f = np.fft.fft2(gray.astype(np.float64))
        f_shift = np.fft.fftshift(f)
        magnitude = np.abs(f_shift)

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4

        y_grid, x_grid = np.ogrid[:h, :w]
        low_mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2
        high_mask = ~low_mask

        total_energy = np.sum(magnitude) + 1e-8
        high_freq_energy = np.sum(magnitude[high_mask]) / total_energy

        return float(high_freq_energy)

    def compute_color_statistics(self, face_rgb):
        """Color channel statistics. GAN faces often have unnatural hue/saturation."""
        hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
        stats = {
            'hue_mean': float(np.mean(hsv[:, :, 0])),
            'hue_std': float(np.std(hsv[:, :, 0])),
            'saturation_mean': float(np.mean(hsv[:, :, 1])),
            'saturation_std': float(np.std(hsv[:, :, 1])),
            'value_mean': float(np.mean(hsv[:, :, 2])),
        }
        return stats

    def compute_edge_consistency(self, gray):
        """Edge density. GAN blending creates unnatural edge patterns at boundaries."""
        edges = cv2.Canny(gray, 50, 150)
        return float(np.sum(edges > 0) / edges.size)

    def compute_compression_artifacts(self, gray):
        """
        Detect block-level JPEG compression artifacts.
        Mismatched compression between face and background is a deepfake indicator.
        """
        # Compute block-wise DCT energy differences (8x8 blocks)
        h, w = gray.shape
        block_energies = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = gray[y:y+8, x:x+8].astype(np.float32) - 128
                dct_block = cv2.dct(block)
                high_freq = dct_block[4:, 4:]
                block_energies.append(float(np.sum(high_freq ** 2)))

        if not block_energies:
            return 0.0
        return float(np.std(block_energies) / (np.mean(block_energies) + 1e-8))

    def analyze_face(self, face_rgb):
        """
        Run full forensic analysis on a single face crop (RGB numpy array).
        
        Returns:
            dict of forensic signal scores
        """
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

        sharpness = self.compute_laplacian_sharpness(gray)
        noise_residual = self.compute_noise_residual(gray)
        high_freq_ratio = self.compute_frequency_artifacts(gray)
        color_stats = self.compute_color_statistics(face_rgb)
        edge_density = self.compute_edge_consistency(gray)
        compression_artifact = self.compute_compression_artifacts(gray)

        # Compute a forensic anomaly score (heuristic baseline)
        # These thresholds derived from empirical analysis of real vs GAN faces
        anomaly_signals = []
        if sharpness < 80:           anomaly_signals.append(0.3)   # over-smooth
        if noise_residual < 2.0:     anomaly_signals.append(0.2)   # low noise
        if high_freq_ratio > 0.85:   anomaly_signals.append(0.25)  # HF artifacts
        if color_stats['hue_std'] < 8: anomaly_signals.append(0.15) # flat hue
        if edge_density < 0.03:      anomaly_signals.append(0.1)   # blurry edges

        forensic_score = float(min(sum(anomaly_signals), 1.0))

        return {
            'sharpness': sharpness,
            'noise_residual': noise_residual,
            'high_freq_ratio': high_freq_ratio,
            'edge_density': edge_density,
            'compression_artifact': compression_artifact,
            'forensic_score': forensic_score,
            **color_stats
        }

    def analyze_batch(self, face_crops):
        """
        Analyze a batch of face crops.
        Returns list of forensic dicts and an aggregated summary.
        """
        results = []
        for crop in face_crops:
            try:
                results.append(self.analyze_face(crop))
            except Exception as e:
                logger.warning(f"Forensic analysis failed for a crop: {e}")
                results.append({'forensic_score': 0.0, 'error': str(e)})

        if results:
            scores = [r.get('forensic_score', 0.0) for r in results]
            summary = {
                'mean_forensic_score': float(np.mean(scores)),
                'max_forensic_score': float(np.max(scores)),
                'min_forensic_score': float(np.min(scores)),
                'std_forensic_score': float(np.std(scores)),
                'suspicious_face_ratio': float(sum(s > 0.3 for s in scores) / len(scores))
            }
        else:
            summary = {'mean_forensic_score': 0.0, 'suspicious_face_ratio': 0.0}

        return results, summary
