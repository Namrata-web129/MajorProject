"""
Face Detection Module
Uses MTCNN for face detection with multi-face support.
Falls back to OpenCV Haar cascades if MTCNN is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias
BBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class DetectedFace:
    """A single detected face with bounding box and optional landmarks."""
    bbox: BBox                        # (x, y, width, height)
    confidence: float
    landmarks: Optional[dict] = None  # e.g. {'left_eye': (x,y), ...}
    face_image: Optional[np.ndarray] = None  # Cropped face (BGR)


class FaceDetector:
    """
    Face detector with MTCNN as the primary backend and OpenCV Haar cascade
    as a fallback.  Supports detecting multiple faces per image.
    """

    def __init__(
        self,
        min_face_size: int = 40,
        confidence_threshold: float = 0.90,
        device: str = "cpu",
    ) -> None:
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._backend: str = "none"
        self._mtcnn = None
        self._haar_cascade = None
        self._init_detector()

    # ── Initialisation ────────────────────────────────────────────────────
    def _init_detector(self) -> None:
        """Try MTCNN first; fall back to OpenCV Haar cascade."""
        try:
            from facenet_pytorch import MTCNN as FacenetMTCNN
            import torch

            device_torch = torch.device(self.device)
            self._mtcnn = FacenetMTCNN(
                image_size=320,
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                keep_all=True,
                device=device_torch,
                post_process=False,
            )
            self._backend = "mtcnn"
            logger.info("Face detector initialised with MTCNN (device=%s)", self.device)
        except ImportError:
            logger.warning("facenet-pytorch not available – falling back to OpenCV Haar cascade")
            self._haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
            )
            self._backend = "haar"

    # ── Public API ────────────────────────────────────────────────────────
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a BGR image.

        Returns a list of DetectedFace objects sorted by confidence (descending).
        """
        if self._backend == "mtcnn":
            return self._detect_mtcnn(image)
        elif self._backend == "haar":
            return self._detect_haar(image)
        else:
            raise RuntimeError("No face detection backend available")

    def detect_and_crop(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (320, 320),
        padding_ratio: float = 0.1,
    ) -> List[DetectedFace]:
        """
        Detect faces and crop + resize each to target_size.
        Adds a margin around the bounding box for context.
        """
        faces = self.detect(image)
        h, w = image.shape[:2]

        for face in faces:
            x, y, fw, fh = face.bbox
            # Add padding
            pad_x = int(fw * padding_ratio)
            pad_y = int(fh * padding_ratio)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            face.face_image = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

        # Filter faces that have a valid crop
        return [f for f in faces if f.face_image is not None]

    # ── MTCNN backend ─────────────────────────────────────────────────────
    def _detect_mtcnn(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using MTCNN from facenet-pytorch."""
        # MTCNN expects RGB PIL or numpy array
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self._mtcnn.detect(rgb, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return []

        faces: List[DetectedFace] = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None or prob < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            lm = None
            if landmarks is not None and i < len(landmarks):
                pts = landmarks[i]
                lm = {
                    "left_eye": tuple(pts[0].astype(int)),
                    "right_eye": tuple(pts[1].astype(int)),
                    "nose": tuple(pts[2].astype(int)),
                    "mouth_left": tuple(pts[3].astype(int)),
                    "mouth_right": tuple(pts[4].astype(int)),
                }
            faces.append(DetectedFace(
                bbox=(x1, y1, x2 - x1, y2 - y1),
                confidence=float(prob),
                landmarks=lm,
            ))

        faces.sort(key=lambda f: f.confidence, reverse=True)
        return faces

    # ── Haar fallback ─────────────────────────────────────────────────────
    def _detect_haar(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using OpenCV Haar cascade (CPU-only fallback)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self._haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if len(rects) == 0:
            return []

        faces: List[DetectedFace] = []
        for x, y, w, h in rects:
            faces.append(DetectedFace(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=0.95,  # Haar doesn't provide confidence
            ))
        return faces

    @property
    def backend_name(self) -> str:
        return self._backend
