from pathlib import Path

from app.models import SubtitleCue
from app.subtitles.srt_exporter import SRTExporter


def test_srt_exporter_sorts_reindexes_and_writes_utf8(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.srt"
    path = SRTExporter().export(
        [
            SubtitleCue(index=9, start=2.0, end=3.25, text="Привіт"),
            SubtitleCue(index=3, start=0.0, end=1.5, text="Hello"),
        ],
        output,
    )

    assert path == output
    assert output.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n"
        "2\n00:00:02,000 --> 00:00:03,250\nПривіт\n\n"
    )
