# txt2imgBench - StreamDiffusion MPS Performance Benchmark

A benchmarking tool for measuring text-to-image generation performance using StreamDiffusion with Hyper-SD LoRA on Apple Silicon (MPS).

## Features

- **Manual StreamDiffusion Setup**: Direct control over pipeline configuration
- **Hyper-SD 1-Step LoRA**: Ultra-fast single-step inference with TCD scheduler
- **MPS Optimization**: Designed for Apple Silicon performance measurement
- **Real-time FPS Tracking**: Precise performance measurement using `time.perf_counter()`
- **Simple UI**: Clean interface focused on benchmarking

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Build frontend:
```bash
npm run build
```

## Usage

1. Start the backend server:
```bash
python main.py
```

2. Open your browser to `http://127.0.0.1:9091`

3. Enter a prompt and either:
   - Generate single images
   - Start continuous benchmarking

## Performance Metrics

- **Current FPS**: Instantaneous frames per second
- **Average FPS**: Rolling average over recent generations
- **Min/Max FPS**: Performance bounds during session
- **Inference Time**: Per-image generation time in milliseconds

## Configuration

Edit `config.py` to modify:

- Model settings (base model, LoRA configuration)
- Image dimensions (512x512 default)
- FPS measurement window size
- Device and optimization settings

## Technical Details

- **Base Model**: Stable Diffusion v1.5
- **LoRA**: Hyper-SD 1-step LoRA from ByteDance
- **Scheduler**: TCD (Trajectory Consistency Distillation)
- **Inference Steps**: 1 (ultra-fast generation)
- **CFG**: Disabled for maximum speed
- **Device**: MPS (Apple Silicon GPU)

This tool provides a baseline for MPS performance measurement and optimization experiments.