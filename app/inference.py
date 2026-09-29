from pathlib import Path

import numpy as np
from PIL import Image


class SiglipBottleEncoder:
    def __init__(
        self,
        model_id: str,
        finetuned_weights: Path,
        bottle_detector_weights: Path,
        label_detector_weights: Path,
        bottle_confidence: float,
        label_confidence: float,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor
        from ultralytics import YOLO

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()

        try:
            checkpoint = torch.load(finetuned_weights, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(finetuned_weights, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.model.load_state_dict(checkpoint, strict=True)

        self.bottle_detector = YOLO(str(bottle_detector_weights)) if bottle_detector_weights.is_file() else None
        self.label_detector = YOLO(str(label_detector_weights)) if label_detector_weights.is_file() else None
        if self.bottle_detector is not None and self.bottle_detector.task != "detect":
            raise ValueError("BOTTLE_DETECTOR_WEIGHTS должен указывать на YOLO detect модель")
        if self.label_detector is not None and self.label_detector.task != "pose":
            raise ValueError("LABEL_DETECTOR_WEIGHTS должен указывать на YOLO pose модель")
        self.bottle_confidence = bottle_confidence
        self.label_confidence = label_confidence

    @staticmethod
    def _ordered_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)[:4]
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Для перспективного кропа этикетки нужны 4 keypoints")
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).ravel()
        ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
        ordered[1], ordered[3] = points[np.argmin(diffs)], points[np.argmax(diffs)]
        return ordered

    def _bottle_crop(self, image_rgb: np.ndarray) -> np.ndarray | None:
        if self.bottle_detector is None:
            return None
        import cv2

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result = self.bottle_detector.predict(
            image_bgr, conf=self.bottle_confidence, verbose=False
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        best = int(result.boxes.conf.argmax())
        x1, y1, x2, y2 = result.boxes.xyxy[best].cpu().numpy().astype(int)
        height, width = image_rgb.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        return image_rgb[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None

    def _label_crop(self, image_rgb: np.ndarray) -> np.ndarray | None:
        if self.label_detector is None:
            return None
        import cv2

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result = self.label_detector.predict(
            image_bgr, conf=self.label_confidence, verbose=False
        )[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return None
        best = int(result.boxes.conf.argmax()) if result.boxes is not None and len(result.boxes) else 0
        try:
            corners = self._ordered_points(result.keypoints.xy[best].cpu().numpy())
        except (ValueError, IndexError):
            return None

        width, height = 400, 600
        destination = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        matrix = cv2.getPerspectiveTransform(corners, destination)
        return cv2.warpPerspective(image_rgb, matrix, (width, height))

    def encode(self, image: Image.Image) -> tuple[np.ndarray, str, bool, bool]:
        original = np.asarray(image.convert("RGB"))
        bottle = self._bottle_crop(original)
        bottle_found = bottle is not None
        bottle_or_image = bottle if bottle_found else original
        label = self._label_crop(bottle_or_image) if bottle_found else None
        label_found = label is not None
        source = label if label_found else bottle_or_image
        mode = "label" if label_found else "full"

        inputs = self.processor(images=source, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.device == "cuda":
            inputs = {
                key: value.half() if value.is_floating_point() else value
                for key, value in inputs.items()
            }
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        if not self.torch.is_tensor(features):
            features = getattr(features, "pooler_output", None)
            if features is None:
                raise RuntimeError("SigLIP не вернул pooled image features")
        if features.ndim == 3:
            features = features.mean(dim=1)
        vector = self.torch.nn.functional.normalize(features.float(), p=2, dim=-1)[0]
        return vector.cpu().numpy(), mode, bottle_found, label_found