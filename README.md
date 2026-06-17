<p align="center">
  <h1 align="center">Padelytics & BandejAI</h1>
  <p align="center"><strong>Open-source, single-camera tactical & statistical analysis for padel</strong></p>
  <p align="center">
    <a href="https://github.com/Moloshow/BandejAI/actions"><img alt="Status" src="https://img.shields.io/badge/status-pre--alpha-orange"></a>
    <a href="https://github.com/Moloshow/BandejAI"><img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue"></a>
    <a href="https://github.com/Moloshow/BandejAI"><img alt="CUDA" src="https://img.shields.io/badge/CUDA-recommended-green"></a>
    <a href="https://github.com/Moloshow/BandejAI/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
    <a href="https://github.com/Moloshow/BandejAI/issues"><img alt="Issues" src="https://img.shields.io/badge/contributions-welcome-brightgreen"></a>
  </p>
</p>

> **Status: Pre-alpha.** This project is under active design. No code is implemented yet. This README describes the intended architecture.

---

## Overview

**Padelytics** is an open-source ecosystem dedicated to tactical and statistical analysis of padel. Its core AI engine, **BandejAI**, extracts advanced metrics from a **single monocular video recording** and turns them into actionable coaching insights.

The MVP runs entirely **offline** (asynchronous batch processing of video files). A multimodal pipeline fuses **computer vision**, **acoustic analysis**, and **local large language models** to deliver deep tactical insights and an automated "Virtual Coach".

---

## System Architecture

BandejAI separates concerns across **three isolated layers**: extraction (deep learning), deterministic geometry/kinematics, and semantic reasoning (multimodal LLM).

```mermaid
graph TD
    Video[Video Match Input]

    Video --> VisionPipeline[Vision Pipeline]
    Video --> AudioPipeline[Audio Pipeline]

    VisionPipeline --> YOLO["YOLOv8<br/>(player tracking)"]
    VisionPipeline --> TrackNet["TrackNetV5<br/>(ball tracking)"]
    VisionPipeline --> PoseC3D["PoseC3D + RGB<br/>(action recognition)"]

    AudioPipeline --> HighPass["High-pass filter<br/>(10 kHz)"]
    HighPass --> EMAPeak["EMA peak detection"]
    EMAPeak --> YAMNet["YAMNet-256<br/>(surface classification)"]

    YOLO --> Kinematics[Homography & Kinematics Engine]
    TrackNet --> Kinematics
    YAMNet --> Kinematics

    Kinematics --> StructJSON["Structured JSON<br/>(Pydantic-validated)"]
    StructJSON --> LLM["Local LLM<br/>(constrained decoding)"]
    LLM --> Output[Padelytics Final Insights]
```

---

## Core Features

### 1. Computer Vision & Deep Learning
- **Player tracking** - Bounding-box detection with YOLOv8 paired with identity tracking (ByteTrack / DeepSORT). High-fidelity skeleton extraction (13 DoF) via ViTPose.
- **Ball tracking** - Spatio-temporal detection using **TrackNetV5**, which ingests a 3-frame sliding window and predicts Gaussian heatmaps instead of bounding boxes. This is critical for surviving motion blur and complex glass/mesh backgrounds.
- **Action recognition** - Fine-grained stroke classification (*Bandeja*, *Vibora*, *Chiquita*) using a **PoseC3D** network enhanced with **RGB Early-Fusion**, capturing both body kinematics and racket-head angles that pure skeleton graphs (ST-GCN) miss.

### 2. Geometry & Kinematics (deterministic)
- **Homography projection** - Transform image coordinates `(u, v)` to 2D court-space `(x, y)` via a perspective matrix computed from court keypoints. Players are projected from the bottom of their bounding box (planar assumption), enabling accurate occupation heatmaps. Ball coordinates are only recorded at certified bounce moments (see audio) to avoid fictitious in-flight projections.

  `[[x], [y], [1]] = c * H * [[u], [v], [1]]`

- **Ball kinematics** - Velocity estimation from projected positions, smoothed by a **Kalman filter** to mitigate tracking jitter and interpolate micro-occlusions.
- **The "Corde"** - Euclidean distance between partners over time, a key tactical indicator of defensive/offensive synchronization. Time-series analysis of this distance quantifies transition discipline (synchronized net approaches).

### 3. Multimodal Analysis & Generative Coaching
- **Acoustic refereeing** - Millisecond-accurate bounce detection using a high-pass Butterworth filter (10 kHz, zero-phase) and EMA energy thresholding. Surface classification (glass vs. turf vs. racket) is handled by a fine-tuned **YAMNet-256** micro-CNN fed 50 ms Mel-spectrograms around each peak - far more reliable than raw YAMNet's 960 ms windows for impulsive 1-5 ms events.
- **Virtual Coach** - 2D game events are translated into structured JSON and processed by a **local LLM** (e.g., Hermes 2 Pro / Qwen-JSON, 7-8B) with **constrained decoding** (Pydantic / JSON Schema), making syntactically invalid output mathematically impossible. Raw geometry is pre-translated into semantic spatial concepts ("transition zone", "net", "back court") before prompting, so the LLM reasons on interpretable context.

---

## Key Technical Decisions

| Domain | Choice | Rationale |
|---|---|---|
| Ball detection | **TrackNetV5** (heatmaps, 3-frame window) | Survives motion blur & glass/mesh backgrounds where YOLO fails |
| Stroke classification | **PoseC3D + RGB Early-Fusion** | Skeleton heatmaps capture biomechanics; RGB adds racket angle/effect - essential to separate *Bandeja* vs *Vibora* |
| Pose estimation | **ViTPose-L** | Vision-Transformer based; superior to convolutional pose estimators on PadelTracker100 |
| Court projection | **Homography (OpenCV)** + planar assumption | Deterministic, explainable; ball only projected at audio-certified bounces |
| Acoustic events | **2-stage: EMA peak + YAMNet-256** | Ball impacts are 1-5 ms; raw YAMNet's 960 ms window drowns the signal |
| Tactical reports | **Local LLM + constrained decoding** | No regex parsing; Pydantic schema enforced at the token-distribution level |
| Transfer learning | **FineBadminton -> Padel** | *Slice smash* biomechanics (decentered hit, wrist whip) share latent structure with *Vibora* |

---

## Project Structure

```text
BandejAI/
├── vision/                         # Image processing & visual neural networks
│   ├── ball_tracking/              # TrackNetV5 detection + 3-frame inference
│   ├── player_tracking/            # YOLOv8 + ByteTrack identity tracking
│   └── action_recognition/         # PoseC3D + RGB Early-Fusion classifier
│
├── audio/                          # Acoustic signal analysis & refereeing
│   ├── peak_detection.py           # Butterworth high-pass + EMA thresholding
│   └── classification.py           # YAMNet-256 surface (glass/turf/racket)
│
├── core_math/                      # Deterministic geometry & kinematics
│   ├── homography/projector.py     # H matrix computation & 2D projection
│   └── kinematics/
│       ├── kalman_filter.py        # Ball trajectory smoothing
│       └── tactical_metrics.py     # Ball speed & the "Corde" metric
│
├── llm_coach/                      # Tactical report generation via local LLM
│   ├── schemas/output_models.py    # Pydantic output validation
│   └── generation/
│       ├── prompt_templates.py     # 2D-metric -> tactical-semantics translation
│       └── orchestrator.py         # Constrained-decoding LLM inference
│
├── utils/                          # Shared utilities
│   ├── video_processor.py          # Read/write/resize video streams
│   └── visualization.py            # Trajectories, 2D mini-map, heatmaps
│
├── notebooks/                      # Research & training experiments
│   ├── transfer_learning_badminton.ipynb
│   └── homography_calibration.ipynb
│
# (Internal specs are kept private; public documentation will live here in the future.)
│
├── main.py                         # Offline orchestrator (CLI)
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites
- **Python 3.10+**
- **CUDA-compatible GPU** strongly recommended (inference of YOLOv8 + TrackNet + PoseC3D on CPU is impractical for full matches)
- **ffmpeg** (audio extraction)

### Setup

> **Recommended: use a dedicated Conda environment.**
> PyTorch with CUDA must be installed first, from its official index,
> otherwise pip will pull the CPU-only build.

```bash
# 1. Clone
git clone https://github.com/Moloshow/BandejAI.git
cd BandejAI

# 2. Create & activate the Conda environment
conda create -n bandejai python=3.10 -y
conda activate bandejai

# 3. Install PyTorch with CUDA (adjust cu121 -> your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install runtime + dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Configure environment
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

### Verify the installation

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
pytest                  # should pass (fast unit tests only)
ruff check .            # should report no issues
```

> Model weights (YOLOv8, TrackNetV5, ViTPose, PoseC3D, YAMNet-256) are not bundled. A download script will be provided in a future release. For now, follow the per-module instructions in `vision/`, `audio/`, and `llm_coach/`.

---

## Quickstart

Run the offline analysis pipeline on a recorded match:

```bash
python main.py --video_path path/to/match.mp4 --output_dir results/
```

The orchestrator will, in sequence:
1. Prompt for **12 court keypoints** (manual homography initialization - see Roadmap Phase 1)
2. Track players & ball, extract poses
3. Extract audio & detect/classify bounces
4. Classify strokes
5. Generate a structured tactical report

Outputs (in `results/`): 2D trajectories, heatmaps, stroke taxonomy, bounce log, and the LLM-generated coaching report (validated JSON + human-readable summary).

---

## Roadmap (MVP)

| Phase | Goal | Status | Key Dependencies / Risks |
|---|---|:---:|---|
| **1. Spatial core & player tracking** | 2D positions, "Corde", basic heatmaps | next up | Homography instability if the camera moves - mitigated by manual 12-point UI at video start |
| **2. Ball kinematic extraction** | Trajectory, speeds, parabola modeling | planned | TrackNet must be trained/adapted to padel glass & mesh; VRAM-intensive alongside YOLO |
| **3. Acoustic refereeing** | ms-accurate bounce detection, surface classification | planned | Strict audio-video timestamp alignment required to trigger ball homography |
| **4. Advanced action recognition** | *Bandeja*, *Vibora*, *Chiquita* classification | planned | Needs a transfer-learned mini-dataset from badminton foundations (FineBadminton) |
| **5. Semantic analysis & LLM Coach** | Automated tactical feedback, highlight segmentation | planned | Output quality depends on the geometry-semantics translation before LLM prompting |

---

## Contributing

TBD

---

## License

Licensed under the **Apache License, Version 2.0** - see [LICENSE](LICENSE).

---

## Acknowledgements & References

This project builds on and adapts concepts from several open-source works and datasets.

### Reference repositories
- **[DS_Padel](https://github.com/AlvaroNovillo/DS_Padel)** - YOLOv8 player detection, TrackNet ball tracking, homography integration
- **[padel_analytics](https://github.com/Joao-M-Silva/padel_analytics)** - 2D court projection, 13-DoF pose classification structures
- **[CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck)** - LLM-based prompt engineering and scouting-report architecture
- **[TrackNetV5-SDK](https://github.com/codelancera-offical/TrackNetV5-SDK)** - Industrial-grade 3-frame sliding-window ball tracking
- **[tennis-tracking](https://github.com/artLabss/tennis-tracking)** / **[TennisProject](https://github.com/yastrebksv/TennisProject)** - Bounce prediction and time-series stroke classification concepts

### Datasets
- **[PadelTracker100](https://zenodo.org/records/14653706)** - ~100k annotated frames from World Padel Tour 2022 finals (ball/player tracking, ViTPose-L, shot events)
- **[FineBadminton](https://arxiv.org/html/2508.07554v1)** & **VideoBadminton** - 20-subcategory stroke taxonomy enabling transfer learning for *Vibora*/*Bandeja*

### Selected research
- SlowFast Networks (FAIR) - dual-pathway video recognition
- ST-GCN - spatio-temporal graph convolution for skeleton action recognition
- PoseC3D / **Gate-Shift-Pose** - skeleton-heatmap + RGB fusion for fine-grained sports actions
- YAMNet (Google AudioSet) - audio event detection foundation
- Spin/impact sound analysis in racket sports (Sony AI, table-tennis spin detection)
- Constrained decoding & structured LLM outputs (RL-Struct, Cohere, awesome-llm-json)

A public bibliography will be provided in a future documentation release.