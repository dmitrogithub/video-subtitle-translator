# Video Subtitle Translator

Локальная CLI-утилита для создания переведённых `.srt`-субтитров из видео.

```text
video.mp4 → FFmpeg → faster-whisper → NLLB → result.uk.srt
```

Видео, извлечённое аудио, транскрипт и субтитры не отправляются в облачный API.
Интернет нужен только при первой установке FFmpeg и при первой загрузке моделей.

## Возможности

- автоматическое определение языка речи через faster-whisper;
- локальный перевод NLLB на `en`, `ru`, `uk`, `de`, `fi`, `fr`, `es` и `pl`;
- SRT с переносами по словам, максимум двумя строками и сохранением таймкодов;
- CPU, NVIDIA CUDA и автоматический выбор устройства для Whisper;
- независимые провайдеры транскрибации и перевода — удобно добавить OpenAI,
  DeepL, SeamlessM4T или другую локальную модель;
- хранение FFmpeg и модельных кэшей в `components/` проекта;
- YAML-настройки, диагностика GPU и тесты без загрузки моделей.

## Требования

- Python 3.11+;
- Windows 10/11 для автоматической установки FFmpeg;
- свободное место: несколько ГБ для Whisper и около 2.5 ГБ для NLLB;
- для CUDA — NVIDIA GPU, драйвер и CUDA-библиотеки из `requirements-cuda.txt`.

## Быстрый старт: CPU

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

При первом запуске `setup` предложит скачать FFmpeg. При первом переводе будут
скачаны выбранная Whisper-модель и NLLB. Это нормально и выполняется один раз.

Если `--output` не указан, рядом с видео появится файл `movie.uk.srt`.

## Быстрый старт: NVIDIA GPU

```powershell
python -m pip install -r requirements-cuda.txt
python -m app.cli diagnose
python -m app.cli translate "C:\Videos\movie.mp4" --target-language uk --config .\config.fast-gpu.example.yaml
```

Профиль использует Whisper `small`, `float16` и batch 16 на GPU. NLLB по
умолчанию переводит на CPU: это надёжно для установленного CPU-only PyTorch и
не замедляет наиболее тяжёлый этап — транскрибацию. Если установлен CUDA PyTorch,
можно добавить `--translation-device cuda`.

## Команды

```powershell
# Проверить FFmpeg, CUDA и модельные кэши
python -m app.cli diagnose

# Установить FFmpeg в components\ без вопроса
python -m app.cli setup --yes

# Перевести с автоматическим определением языка
python -m app.cli translate "C:\Videos\movie.mp4" --target-language uk

# Явно указать входной язык, путь и CPU-параметры
python -m app.cli translate "C:\Videos\movie.mp4" --source-language en --target-language ru --output "C:\Videos\movie.ru.srt" --device cpu --compute-type int8 --whisper-model small

# Полная справка
python -m app.cli translate --help
```

Главные параметры: `--output`, `--source-language`, `--target-language`,
`--whisper-model`, `--device`, `--compute-type`, `--batch-size`,
`--translation-device`, `--config`, `--keep-temp` и `--verbose`.

## Компоненты и кэши

Программа не использует FFmpeg из системного `PATH`. Все крупные компоненты
находятся в папке проекта:

```text
components/
├── ffmpeg.exe
├── ffprobe.exe
└── models/
    ├── faster-whisper/
    └── nllb/
```

Содержимое `components/`, видео, WAV и SRT исключены из Git через `.gitignore`.
Не добавляйте их в репозиторий.

## Конфигурация

Скопируйте [config.example.yaml](config.example.yaml), измените значения и
передайте файл через `--config`:

```powershell
Copy-Item .\config.example.yaml .\config.yaml
python -m app.cli translate "C:\Videos\movie.mp4" --target-language uk --config .\config.yaml
```

Приоритет настроек: аргументы CLI → YAML → встроенные значения по умолчанию.

Готовые профили:

- [config.fast-cpu.example.yaml](config.fast-cpu.example.yaml) — `small` + `int8`;
- [config.fast-gpu.example.yaml](config.fast-gpu.example.yaml) — `small` + CUDA + batch 16.

## Архитектура

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

## Разработка

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Тесты используют mock-провайдеры и не скачивают реальные Whisper/NLLB-модели.

## Ограничения

- автоматические таймкоды и машинный перевод стоит проверять перед публикацией;
- на длинном видео CPU-обработка может занимать значительное время;
- очень короткий блок или сверхдлинное слово не всегда позволяют одновременно
  выполнить все правила читабельности и сохранить исходный временной диапазон;
- перед коммерческим или публичным распространением проверьте лицензию NLLB и
  условия использования модели.

## Лицензия

Лицензия пока не выбрана. Перед публикацией установите подходящую лицензию
(`MIT`, `Apache-2.0`, `GPL-3.0` и т. п.) и добавьте файл `LICENSE`.
