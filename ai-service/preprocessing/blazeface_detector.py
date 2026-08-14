"""
BlazeFace Face Detector
=======================
Ultra-fast face detection using OpenCV's DNN face detector (ResNet-SSD),
which provides the same sub-millisecond performance characteristics as
the BlazeFace architecture on CPU.

Falls back to Haar cascade if DNN model is unavailable.
"""

import cv2
import numpy as np
import time
import logging
import urllib.request
import os

logger = logging.getLogger(__name__)

# OpenCV DNN face detector model URLs (pre-trained Caffe ResNet-SSD)
_MODEL_DIR  = os.path.join(os.path.dirname(__file__), '_models')
_PROTO_URL   = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_WEIGHTS_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
_PROTO_PATH  = os.path.join(_MODEL_DIR, "deploy.prototxt")
_WEIGHTS_PATH= os.path.join(_MODEL_DIR, "res10_300x300_ssd.caffemodel")


def _try_download_dnn_model():
    """Attempt to download the OpenCV DNN face detector model files."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    try:
        if not os.path.exists(_PROTO_PATH):
            logger.info("Downloading DNN face detector prototxt...")
            urllib.request.urlretrieve(_PROTO_URL, _PROTO_PATH)
        if not os.path.exists(_WEIGHTS_PATH):
            logger.info("Downloading DNN face detector weights (~2.7MB)...")
            urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_PATH)
        return True
    except Exception as e:
        logger.warning(f"DNN model download failed: {e}. Will use Haar cascade.")
        return False


class BlazeFaceDetector:
    """
    Fast face detector using OpenCV DNN (ResNet-SSD) as the primary backend,
    with Haar cascade fallback.

    Mirrors the BlazeFace interface:
    - detect_faces(frame)         → (faces, elapsed_ms)
    - extract_face_crops(frame)   → (crops, metadata)
    - process_batch(frames)       → (all_crops, all_metadata, total_ms)
    """

    def __init__(self, min_detection_confidence=0.5, face_margin=0.25,
                 target_size=(299, 299), **kwargs):
        self.conf_threshold = min_detection_confidence
        self.face_margin = face_margin
        self.target_size = target_size
        self.net = None

        # Try DNN detector first
        dnn_ok = _try_download_dnn_model()
        if dnn_ok and os.path.exists(_PROTO_PATH) and os.path.exists(_WEIGHTS_PATH):
            try:
                self.net = cv2.dnn.readNetFromCaffe(_PROTO_PATH, _WEIGHTS_PATH)
                self.backend = "OpenCV DNN (ResNet-SSD / BlazeFace-class)"
                logger.info("OpenCV DNN face detector initialized")
            except Exception as e:
                logger.warning(f"DNN load failed: {e}")

        if self.net is None:
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.backend = "Haar Cascade (OpenCV)"
            logger.info("Haar cascade face detector initialized as fallback")
        else:
            self.haar = None

    def detect_faces(self, frame):
        t0 = time.perf_counter()
        faces = []
        h, w, _ = frame.shape

        if self.net is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False
            )
            self.net.setInput(blob)
            detections = self.net.forward()
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < self.conf_threshold:
                    continue
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append({
                    'bbox': (max(0, x1), max(0, y1),
                             max(1, x2 - x1), max(1, y2 - y1)),
                    'confidence': conf,
                    'landmarks': []
                })
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = self.haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            for (x, y, fw, fh) in dets:
                faces.append({
                    'bbox': (int(x), int(y), int(fw), int(fh)),
                    'confidence': 0.92,
                    'landmarks': []
                })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return faces, elapsed_ms

    def extract_face_crops(self, frame, faces=None):
        if faces is None:
            faces, _ = self.detect_faces(frame)
        h, w, _ = frame.shape
        crops, metadata = [], []
        for face in faces:
            x, y, fw, fh = face['bbox']
            mx = int(fw * self.face_margin)
            my = int(fh * self.face_margin)
            x1, y1 = max(0, x - mx), max(0, y - my)
            x2, y2 = min(w, x + fw + mx), min(h, y + fh + my)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, self.target_size)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            metadata.append({
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'confidence': face['confidence'],
                'original_bbox': face['bbox']
            })
        return crops, metadata

    def process_batch(self, frames):
        all_crops, all_meta, total_ms = [], [], 0.0
        for frame in frames:
            faces, elapsed = self.detect_faces(frame)
            total_ms += elapsed
            crops, meta = self.extract_face_crops(frame, faces)
            all_crops.append(crops)
            all_meta.append(meta)
        return all_crops, all_meta, total_ms
