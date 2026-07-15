"""Readable, time-preserving subtitle cue formatting."""

from __future__ import annotations

from math import ceil

from app.models import SubtitleCue, SubtitleSettings, TranslationBlock


class SubtitleFormatter:
    """Split translated blocks into at most two readable lines per cue."""

    def __init__(self, settings: SubtitleSettings) -> None:
        self.settings = settings

    def _line_wrap(self, words: list[str]) -> list[str]:
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if current and len(candidate) > self.settings.max_chars_per_line:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return lines

    def _render_words(self, words: list[str]) -> str:
        return "\n".join(self._line_wrap(words))

    def _initial_pieces(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        lines = self._line_wrap(words)
        return [
            "\n".join(lines[index : index + self.settings.max_lines])
            for index in range(0, len(lines), self.settings.max_lines)
        ]

    @staticmethod
    def _split_index(words: list[str]) -> int:
        middle = len(words) // 2
        punctuation_breaks = [
            index
            for index, word in enumerate(words[:-1], start=1)
            if word.rstrip().endswith((",", ";", ":", ".", "!", "?", "…"))
        ]
        if punctuation_breaks:
            return min(punctuation_breaks, key=lambda index: abs(index - middle))
        return middle

    def _split_to_count(self, pieces: list[str], desired: int) -> list[str]:
        result = list(pieces)
        while len(result) < desired:
            candidates = [
                (len(piece.replace("\n", " ")), index, piece.split())
                for index, piece in enumerate(result)
                if len(piece.split()) > 1
            ]
            if not candidates:
                break
            _, index, words = max(candidates)
            split_at = self._split_index(words)
            result[index : index + 1] = [
                self._render_words(words[:split_at]),
                self._render_words(words[split_at:]),
            ]
        return result

    def _desired_cue_count(self, pieces: list[str], duration: float) -> int:
        characters = sum(len(piece.replace("\n", "")) for piece in pieces)
        duration_count = ceil(duration / self.settings.max_duration)
        speed_count = ceil(
            characters / (self.settings.max_chars_per_second * self.settings.max_duration)
        )
        return max(len(pieces), duration_count, speed_count)

    def _durations(self, pieces: list[str], duration: float) -> list[float]:
        """Allocate a source interval by text length within feasible time bounds."""
        count = len(pieces)
        if count == 0 or duration <= 0:
            return []
        lower = self.settings.min_duration if duration >= count * self.settings.min_duration else 0.0
        upper = self.settings.max_duration if duration <= count * self.settings.max_duration else float("inf")
        weights = [max(1, len(piece.replace("\n", ""))) for piece in pieces]
        remaining_duration = duration
        active = set(range(count))
        result = [0.0] * count
        while active:
            weight_total = sum(weights[index] for index in active)
            proposed = {
                index: remaining_duration * weights[index] / weight_total
                for index in active
            }
            too_short = [index for index, value in proposed.items() if value < lower]
            too_long = [index for index, value in proposed.items() if value > upper]
            if not too_short and not too_long:
                for index, value in proposed.items():
                    result[index] = value
                break
            for index in too_short:
                result[index] = lower
                remaining_duration -= lower
                active.remove(index)
            for index in too_long:
                if index in active:
                    result[index] = upper
                    remaining_duration -= upper
                    active.remove(index)
        result[-1] += duration - sum(result)
        return result

    def format_blocks(self, blocks: list[TranslationBlock]) -> list[SubtitleCue]:
        """Format translated blocks without moving their overall time ranges."""
        cues: list[SubtitleCue] = []
        for block in sorted(blocks, key=lambda item: (item.start, item.id)):
            if block.end <= block.start or not block.translated_text or not block.translated_text.strip():
                continue
            pieces = self._initial_pieces(block.translated_text)
            pieces = self._split_to_count(
                pieces,
                self._desired_cue_count(pieces, block.end - block.start),
            )
            durations = self._durations(pieces, block.end - block.start)
            cursor = block.start
            for position, (piece, duration) in enumerate(zip(pieces, durations, strict=True)):
                end = block.end if position == len(pieces) - 1 else cursor + duration
                if end > cursor and piece.strip():
                    cues.append(
                        SubtitleCue(
                            index=len(cues) + 1,
                            start=cursor,
                            end=end,
                            text=piece,
                        )
                    )
                cursor = end
        return cues
