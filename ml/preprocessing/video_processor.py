"""
Video Processing Module
Frame extraction from video files using OpenCV.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    """A single extracted video frame."""
    frame_index: int           # 0-based index in extracted sequence
    original_frame_number: int # Original frame number in the video
    timestamp_seconds: float   # Timestamp in seconds
    image: np.ndarray          # BGR numpy array


class VideoProcessor:
    """
    Extract frames from video files with configurable sampling.

    Supports MP4, AVI, MOV formats via OpenCV's VideoCapture.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        sample_rate: int = 1,    # Extract every Nth frame (1 = every frame)
        max_frames: int = 50,     # Maximum frames to extract
        timeout_seconds: int = 300,
    ) -> None:
        self.sample_rate = max(1, sample_rate)
        self.max_frames = max_frames
        self.timeout_seconds = timeout_seconds

    def validate_video(self, video_path: str) -> dict:
        """
        Validate a video file and return its properties.
        Raises ValueError if the video cannot be opened.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{ext}'. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0.0

            return {
                "path": str(path),
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration_seconds": duration,
                "file_size_bytes": path.stat().st_size,
            }
        finally:
            cap.release()

    def extract_frames(
        self,
        video_path: str,
        sample_rate: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> List[ExtractedFrame]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file
            sample_rate: Override default sample rate (every Nth frame)
            max_frames: Override maximum number of frames

        Returns:
            List of ExtractedFrame objects
        """
        rate = sample_rate or self.sample_rate
        limit = max_frames or self.max_frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List[ExtractedFrame] = []
        frame_number = 0
        extracted_count = 0

        try:
            while cap.isOpened() and extracted_count < limit:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % rate == 0:
                    timestamp = frame_number / fps
                    frames.append(ExtractedFrame(
                        frame_index=extracted_count,
                        original_frame_number=frame_number,
                        timestamp_seconds=round(timestamp, 3),
                        image=frame,
                    ))
                    extracted_count += 1

                frame_number += 1
        finally:
            cap.release()

        logger.info(
            "Extracted %d frames from %s (total: %d, sample_rate: %d)",
            len(frames), video_path, frame_number, rate,
        )
        return frames

    def extract_frames_at_timestamps(
        self,
        video_path: str,
        timestamps: List[float],
    ) -> List[ExtractedFrame]:
        """Extract frames at specific timestamps (seconds)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List[ExtractedFrame] = []

        try:
            for idx, ts in enumerate(timestamps):
                frame_num = int(ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    frames.append(ExtractedFrame(
                        frame_index=idx,
                        original_frame_number=frame_num,
                        timestamp_seconds=ts,
                        image=frame,
                    ))
        finally:
            cap.release()

        return frames
