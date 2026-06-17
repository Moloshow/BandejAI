"""BandejAI global configuration.

All settings are loaded from environment variables (see `.env.example`).
This centralizes reproducibility (SEED) and device management.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        seed: Global random seed for reproducibility.
        device: Compute device ("cuda" or "cpu").
        data_dir: Base directory for input videos.
        output_dir: Base directory for generated results.
        models_dir: Directory for pretrained model weights.
        yolo_weights: YOLOv8 weights filename for player detection.
        tracknet_weights: TrackNetV5 weights filename for ball tracking.
        vitpose_config: ViTPose config filename.
        vitpose_weights: ViTPose weights filename.
        audio_highpass_hz: Butterworth high-pass filter cutoff frequency.
        llm_model: Path to local quantized LLM (GGUF format).
        llm_max_tokens: Maximum number of tokens to generate.

    """

    model_config = SettingsConfigDict(
        env_prefix="BANDEJAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime ---
    seed: int = 42
    device: str = "cuda"

    # --- Paths ---
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./results")
    models_dir: Path = Path("./models")

    # --- Models ---
    yolo_weights: str = "yolov8x.pt"
    tracknet_weights: str = "TrackNetV5.pt"
    vitpose_config: str = "vitpose_large_coco_256x192.py"
    vitpose_weights: str = "vitpose_large_coco_256x192.pth"

    # --- Audio ---
    audio_highpass_hz: int = 10000

    # --- LLM Coach ---
    llm_model: Path = Path("./models/hermes-2-pro-mistral-7b.Q4_K_M.gguf")
    llm_max_tokens: int = 2048

    @property
    def torch_device(self) -> torch.device:
        """Resolve the compute device as a torch.device object.

        Returns:
            Configured torch device (cuda if available and requested, else cpu).

        """
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def apply_seed(self) -> None:
        """Seed all random number generators for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


# Singleton instance
settings = Settings()
