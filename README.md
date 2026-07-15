# Video Subtitle Translator

A local command-line tool that generates translated `.srt` subtitles from a
video file.

```text
video.mp4 → FFmpeg → faster-whisper → NLLB → result.uk.srt
```

Videos, extracted audio, transcripts, and subtitles are never sent to a cloud
API. Internet access is needed only the first time FFmpeg or a model is
downloaded.

## Features

- automatic speech-language detection with faster-whisper;
- local NLLB translation for `en`, `ru`, `uk`, `de`, `fi`, `fr`, `es`, and `pl`;
- readable SRT output: word-aware wrapping, up to two lines per cue, and
  preserved source timing;
- CPU, NVIDIA CUDA, and automatic device selection for Whisper;
- replaceable transcription and translation providers, ready for OpenAI, DeepL,
  SeamlessM4T, or other local models;
- project-local FFmpeg and model caches under `components/`;
- YAML configuration, GPU diagnostics, and tests that do not download models.

## Requirements

- Python 3.11 or newer;
- Windows 10/11 for automatic FFmpeg installation;
- several GB of free storage for Whisper and approximately 2.5 GB for NLLB;
- an NVIDIA GPU, driver, and the CUDA dependencies in `requirements-cuda.txt`
  for GPU acceleration.

## Quick start: CPU

```powershell
git clone <YOUR_REPOSITORY_URL>
cd video-subtitle-translator

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m app.cli setup
python -m app.cli translate "C:\Videos\movie.mp4" --target-language uk --config .\config.fast-cpu.example.yaml
```

On the first run, `setup` offers to download FFmpeg. The first translation also
downloads the selected Whisper model and NLLB model. These downloads happen once
and are reused later.

When `--output` is omitted, `movie.mp4` produces `movie.uk.srt` next to the
video.

## Quick start: NVIDIA GPU

```powershell
python -m pip install -r requirements-cuda.txt
python -m app.cli diagnose
python -m app.cli translate "C:\Videos\movie.mp4" --target-language uk --config .\config.fast-gpu.example.yaml
```

The GPU profile uses Whisper `small`, `float16`, and batch size 16. NLLB uses
the CPU by default because a standard PyTorch installation can be CPU-only. This
does not slow down the heaviest stage, speech transcription. If CUDA PyTorch is
installed later, add `--translation-device cuda`.

## Commands

```powershell
# Inspect FFmpeg, CUDA, and local model caches
python -m app.cli diagnose

# Install FFmpeg into components\ without a confirmation prompt
python -m app.cli setup --yes

# Translate with automatic source-language detection
python -m app.cli translate "C:\Videos\movie.mp4" --target-language uk

# Set the source language, output path, and CPU options explicitly
python -m app.cli translate "C:\Videos\movie.mp4" --source-language en --target-language ru --output "C:\Videos\movie.ru.srt" --device cpu --compute-type int8 --whisper-model small

# Show all options
python -m app.cli translate --help
```

The main options are `--output`, `--source-language`, `--target-language`,
`--whisper-model`, `--device`, `--compute-type`, `--batch-size`,
`--translation-device`, `--config`, `--keep-temp`, and `--verbose`.

## Components and caches

The application does not use FFmpeg from the system `PATH`. All large local
components are stored inside the project:

```text
components/
├── ffmpeg.exe
├── ffprobe.exe
└── models/
    ├── faster-whisper/
    └── nllb/
```

The contents of `components/`, videos, WAV files, and generated SRT files are
excluded from Git through `.gitignore`. Do not commit them.

## Configuration

Copy [config.example.yaml](config.example.yaml), edit the values, and pass it
using `--config`:

```powershell
Copy-Item .\config.example.yaml .\config.yaml
python -m app.cli translate "C:\Videos\movie.mp4" --target-language uk --config .\config.yaml
```

Configuration precedence is: command-line arguments → YAML file → built-in
defaults.

Included profiles:

- [config.fast-cpu.example.yaml](config.fast-cpu.example.yaml) — `small` + `int8`;
- [config.fast-gpu.example.yaml](config.fast-gpu.example.yaml) — `small` + CUDA + batch size 16.

## Architecture

```text
app/
├── audio/extractor.py
├── transcription/
│   ├── base.py
│   └── faster_whisper_provider.py
├── translation/
│   ├── base.py
│   └── nllb_provider.py
├── subtitles/
│   ├── segment_merger.py
│   ├── formatter.py
│   └── srt_exporter.py
├── components.py
├── pipeline.py
└── cli.py
```

To add OpenAI, DeepL, a local LLM, Whisper translate, or SeamlessM4T, implement
the relevant provider contract. `NLLBTranslationProvider` preserves a block's
identifier, source text, source segment IDs, and timing; it only fills
`translated_text`.

## Development

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use mock providers and never download real Whisper or NLLB models.

## Limitations

- Review automatic timestamps and machine translation before publishing.
- CPU processing can take a considerable amount of time for long videos.
- A very short block or a single extremely long word cannot always satisfy every
  readability rule while preserving the original timing.
- Verify the NLLB license and its terms of use before commercial or public
  distribution.

## License

No license has been selected yet. Before publishing the repository, choose a
license such as `MIT`, `Apache-2.0`, or `GPL-3.0` and add a `LICENSE` file.
