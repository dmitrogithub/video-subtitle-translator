"""SRT serialization for subtitle cues."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import srt

from app.models import SubtitleCue


class SRTExporter:
    """Write valid UTF-8 SRT files, sorting and reindexing every cue."""

    def export(self, cues: list[SubtitleCue], output_path: Path) -> Path:
        """Save cues to ``output_path`` and return that path."""
        ordered = sorted(
            (cue for cue in cues if cue.end > cue.start and cue.text.strip()),
            key=lambda cue: (cue.start, cue.end, cue.index),
        )
        subtitles = [
            srt.Subtitle(
                index=index,
                start=timedelta(seconds=cue.start),
                end=timedelta(seconds=cue.end),
                content=cue.text,
            )
            for index, cue in enumerate(ordered, start=1)
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(srt.compose(subtitles, reindex=False), encoding="utf-8")
        return output_path
