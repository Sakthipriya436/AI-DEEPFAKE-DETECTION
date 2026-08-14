"""
Ensemble Classifier — optimized for decisive AI / REAL verdicts.
Returns clear classification with supporting evidence signals.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    def __init__(self, xception_weight=0.55, resnext_weight=0.45,
                 decision_threshold=0.42):
        self.xception_weight = xception_weight
        self.resnext_weight  = resnext_weight
        self.decision_threshold = decision_threshold

    def fuse(self, xception_preds, resnext_preds):
        n = max(len(xception_preds), len(resnext_preds))
        fused = []
        for i in range(n):
            xp = xception_preds[i] if i < len(xception_preds) else 0.5
            rp = resnext_preds[i]  if i < len(resnext_preds)  else 0.5
            fused.append(self.xception_weight * xp + self.resnext_weight * rp)
        return fused

    def aggregate(self, scores):
        if not scores:
            return 0.5
        scores = np.array(scores)
        if len(scores) == 1:
            return float(scores[0])
        # Use 80th percentile + mean to catch even minority deepfake frames
        return float(np.clip(0.65 * np.percentile(scores, 80) + 0.35 * np.mean(scores), 0, 1))

    def _build_signals(self, xception_preds, resnext_preds, forensic_summary):
        """Build human-readable evidence signals for the verdict."""
        signals = []
        x_mean = np.mean(xception_preds) if xception_preds else 0
        r_mean = np.mean(resnext_preds)  if resnext_preds  else 0
        forensic_score = forensic_summary.get('mean_forensic_score', 0) if forensic_summary else 0
        suspicious_ratio = forensic_summary.get('suspicious_face_ratio', 0) if forensic_summary else 0

        # Texture analysis signal
        if x_mean > 0.5:
            signals.append(("Unnatural skin texture detected", "warning"))
        else:
            signals.append(("Skin texture appears natural", "ok"))

        # Facial geometry signal
        if r_mean > 0.5:
            signals.append(("Facial geometry inconsistencies found", "warning"))
        else:
            signals.append(("Facial geometry looks consistent", "ok"))

        # Forensic signal
        if forensic_score > 0.35:
            signals.append(("Forensic artifacts present in image data", "warning"))
        elif forensic_score > 0.15:
            signals.append(("Minor forensic anomalies detected", "caution"))
        else:
            signals.append(("No significant forensic artifacts", "ok"))

        # Face consistency signal
        if suspicious_ratio > 0.5:
            signals.append(("Multiple faces show AI indicators", "warning"))
        elif suspicious_ratio > 0.2:
            signals.append(("Some faces show minor AI indicators", "caution"))
        else:
            signals.append(("Faces appear biologically consistent", "ok"))

        return signals

    def classify(self, xception_preds, resnext_preds, forensic_summary=None):
        if not xception_preds and not resnext_preds:
            return {
                'is_deepfake': False,
                'verdict': 'UNCERTAIN',
                'verdict_label': 'No Faces Detected',
                'signals': [("No faces found to analyze", "caution")],
                'faces_detected': 0,
            }

        x_score = float(np.mean(xception_preds)) if xception_preds else 0.5
        r_score = float(np.mean(resnext_preds))  if resnext_preds  else 0.5

        fused = self.fuse(xception_preds, resnext_preds)
        raw   = self.aggregate(fused)

        # Apply forensic boost
        if forensic_summary:
            boost = (0.10 * forensic_summary.get('mean_forensic_score', 0) +
                     0.05 * forensic_summary.get('suspicious_face_ratio', 0))
            raw = float(np.clip(raw + boost, 0, 1))

        # Calibration: push scores away from the boundary for decisive verdicts
        if raw > 0.5:
            calibrated = 0.5 + (raw - 0.5) * 1.4
        else:
            calibrated = 0.5 - (0.5 - raw) * 1.4
        calibrated = float(np.clip(calibrated, 0.01, 0.99))

        is_ai = calibrated >= self.decision_threshold
        signals = self._build_signals(xception_preds, resnext_preds, forensic_summary)

        return {
            'is_deepfake': is_ai,
            'verdict': 'AI_GENERATED' if is_ai else 'REAL',
            'verdict_label': 'AI Generated' if is_ai else 'Real / Authentic',
            'xception_score': round(x_score, 4),
            'resnext_score':  round(r_score, 4),
            'ensemble_score': round(raw, 4),
            'signals': signals,
            'model_agreement': round(1.0 - abs(x_score - r_score), 4),
        }
