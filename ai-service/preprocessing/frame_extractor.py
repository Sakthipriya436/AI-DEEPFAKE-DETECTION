"""
Video Frame Extractor - Efficient frame extraction with adaptive sampling.
Optimized for large video files (100MB+).
"""
import cv2
import numpy as np
import time
import os
import logging

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    def __init__(self, max_frames=32, target_fps=1.0, resize_width=None):
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.resize_width = resize_width

    def get_video_metadata(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        metadata['duration'] = metadata['frame_count'] / max(metadata['fps'], 1)
        metadata['file_size_mb'] = os.path.getsize(video_path) / (1024 * 1024)
        cap.release()
        return metadata

    def compute_sample_indices(self, total_frames, video_fps):
        duration = total_frames / max(video_fps, 1)
        if duration <= 0:
            return [0]
        desired_count = min(int(duration * self.target_fps), self.max_frames)
        desired_count = max(desired_count, 1)
        if duration < 3:
            desired_count = min(total_frames, self.max_frames)
        if desired_count >= total_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / desired_count
            indices = [int(i * step) for i in range(desired_count)]
        if 0 not in indices:
            indices.insert(0, 0)
        if total_frames - 1 not in indices and total_frames > 1:
            indices.append(total_frames - 1)
        return sorted(set(indices))

    def extract_frames(self, video_path):
        start_time = time.perf_counter()
        metadata = self.get_video_metadata(video_path)
        logger.info(f"Video: {metadata['width']}x{metadata['height']}, "
                     f"{metadata['fps']:.1f}fps, {metadata['duration']:.1f}s, "
                     f"{metadata['file_size_mb']:.1f}MB")
        sample_indices = self.compute_sample_indices(metadata['frame_count'], metadata['fps'])
        logger.info(f"Sampling {len(sample_indices)} frames from {metadata['frame_count']} total")
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_indices = []
        sample_set = set(sample_indices)
        current_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in sample_set:
                if self.resize_width and frame.shape[1] > self.resize_width:
                    scale = self.resize_width / frame.shape[1]
                    new_h = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (self.resize_width, new_h))
                frames.append(frame)
                frame_indices.append(current_idx)
            current_idx += 1
            if current_idx > max(sample_indices):
                break
        cap.release()
        extraction_time_ms = (time.perf_counter() - start_time) * 1000
        metadata['frames_extracted'] = len(frames)
        metadata['sample_indices'] = frame_indices
        logger.info(f"Extracted {len(frames)} frames in {extraction_time_ms:.0f}ms")
        return frames, metadata, extraction_time_ms
