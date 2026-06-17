# BandejAI Examples

This directory contains lightweight scripts and snippets that demonstrate how to use BandejAI's internal classes programmatically. These are not part of the core pipeline orchestrator (`main.py`), but serve as developer documentation.

## Available Examples

### 1. `api_tracking_example.py`
A minimal, 40-line script showing how to instantiate the `PlayerTracker` class and run it frame-by-frame on a video without any UI or pipeline overhead.
```bash
python examples/api_tracking_example.py
```

### 2. `demo_scene_detection.py`
A standalone script that uses `PySceneDetect` to parse a full broadcast video and extract pure gameplay rallies (filtering out replays, zoom-ins, and dead time). It prints out the exact frame intervals for each rally, which can then be injected into your `config.yaml`.
```bash
python examples/demo_scene_detection.py --video data/sample_match.mp4.webm
```
