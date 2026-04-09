from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi


@dataclass
class Chapter:
    start_time: float
    end_time: float
    chapter_title: str


@dataclass
class TranscriptSegment:
    start_time: float
    end_time: float
    text: str


@dataclass
class SentenceUnit:
    start_time: float
    end_time: float
    text: str


@dataclass
class Chunk:
    chunk_id: str
    video_id: str
    chapter_title: str
    start_time: float
    end_time: float
    transcript: str


def apply_yt_auth_opts(
    opts: dict[str, Any],
    cookies_file: Path | None,
    cookies_from_browser: str | None,
) -> dict[str, Any]:
    """Attach optional YouTube authentication options for yt-dlp."""
    if cookies_file is not None:
        opts["cookiefile"] = str(cookies_file)
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


class MetadataResolver:
    """Resolve semantic video IDs from playlist metadata."""

    def __init__(self, metadata_path: Path | None) -> None:
        self.metadata_path = metadata_path
        self._by_youtube_id: dict[str, str] = {}
        if metadata_path is not None:
            self._load()

    def _load(self) -> None:
        if self.metadata_path is None:
            return
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)

        if not isinstance(rows, list):
            raise ValueError("Metadata must be a JSON array.")

        for row in rows:
            if not isinstance(row, dict):
                continue
            yt_id = str(row.get("video_id") or "").strip()
            semantic_id = str(row.get("semantic_video_id") or "").strip()
            if yt_id and semantic_id:
                self._by_youtube_id[yt_id] = semantic_id

    def resolve_semantic_id(self, youtube_video_id: str) -> str:
        return self._by_youtube_id.get(youtube_video_id, youtube_video_id)


class YouTubeMetadataError(RuntimeError):
    pass


class TranscriptError(RuntimeError):
    pass


class ChapterExtractor:
    """Extract chapter boundaries from YouTube metadata or description timestamps."""

    DESCRIPTION_TS_RE = re.compile(
        r"^\s*(?P<ts>(?:\d+:)?\d{1,2}:\d{2})\s*(?:-|\||-|–|—)?\s*(?P<title>.+?)\s*$"
    )

    def __init__(self, cookies_file: Path | None, cookies_from_browser: str | None) -> None:
        self._ydl_opts: dict[str, Any] = apply_yt_auth_opts({
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }, cookies_file=cookies_file, cookies_from_browser=cookies_from_browser)

    def fetch_info(self, url: str) -> dict[str, Any]:
        try:
            with yt_dlp.YoutubeDL(self._ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            if not info:
                raise YouTubeMetadataError("Empty metadata response from yt-dlp.")
            return info
        except Exception as exc:
            raise YouTubeMetadataError(f"Failed to extract YouTube metadata: {exc}") from exc

    @staticmethod
    def _to_seconds(ts: str) -> float:
        parts = [int(p) for p in ts.split(":")]
        if len(parts) == 2:
            mm, ss = parts
            return float(mm * 60 + ss)
        hh, mm, ss = parts
        return float(hh * 3600 + mm * 60 + ss)

    def _parse_description_toc(self, description: str, duration: float) -> list[Chapter]:
        entries: list[tuple[float, str]] = []
        for line in description.splitlines():
            match = self.DESCRIPTION_TS_RE.match(line)
            if not match:
                continue
            start_time = self._to_seconds(match.group("ts"))
            title = match.group("title").strip()
            entries.append((start_time, title))

        # Keep unique timestamps in ascending order.
        deduped = sorted({start: title for start, title in entries}.items(), key=lambda x: x[0])
        if len(deduped) < 2 and not (len(deduped) == 1 and deduped[0][0] == 0.0):
            return []

        chapters: list[Chapter] = []
        for idx, (start, title) in enumerate(deduped):
            end = deduped[idx + 1][0] if idx + 1 < len(deduped) else duration
            if end <= start:
                continue
            chapters.append(Chapter(start_time=start, end_time=end, chapter_title=title or f"Chapter {idx + 1}"))
        return chapters

    def extract_chapters(self, info: dict[str, Any]) -> list[Chapter]:
        duration = float(info.get("duration") or 0.0)
        if duration <= 0:
            raise YouTubeMetadataError("Video duration is missing from metadata.")

        raw_chapters = info.get("chapters") or []
        chapters: list[Chapter] = []
        for idx, item in enumerate(raw_chapters):
            start = float(item.get("start_time", 0.0))
            end = float(item.get("end_time", duration))
            title = str(item.get("title") or f"Chapter {idx + 1}").strip()
            if end > start:
                chapters.append(Chapter(start_time=start, end_time=end, chapter_title=title))

        if chapters:
            return chapters

        description = str(info.get("description") or "")
        chapters = self._parse_description_toc(description, duration)
        if chapters:
            return chapters

        return [Chapter(start_time=0.0, end_time=duration, chapter_title="Full Video")]


class TranscriptFetcher:
    """Fetch English transcript using youtube-transcript-api."""

    def fetch(self, video_id: str) -> list[TranscriptSegment]:
        try:
            api = YouTubeTranscriptApi()
            if hasattr(api, "fetch"):
                rows = api.fetch(video_id, languages=["en"])
            else:
                rows = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            segments: list[TranscriptSegment] = []
            for row in rows:
                if isinstance(row, dict):
                    start = float(row.get("start", 0.0))
                    duration = float(row.get("duration", 0.0))
                    text = str(row.get("text", "")).replace("\n", " ").strip()
                else:
                    start = float(getattr(row, "start", 0.0))
                    duration = float(getattr(row, "duration", 0.0))
                    text = str(getattr(row, "text", "")).replace("\n", " ").strip()
                if not text:
                    continue
                segments.append(TranscriptSegment(start_time=start, end_time=start + duration, text=text))
            if not segments:
                raise TranscriptError("Transcript API returned no usable English segments.")
            return segments
        except Exception as exc:
            raise TranscriptError(f"Failed to fetch transcript for {video_id}: {exc}") from exc


class SemanticChunker:
    """Assign transcript into chapter-aware chunks without cutting sentence boundaries."""

    SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*$")

    def __init__(self, max_chunk_seconds: float = 60.0) -> None:
        self.max_chunk_seconds = max_chunk_seconds

    @staticmethod
    def _segment_midpoint(segment: TranscriptSegment) -> float:
        return (segment.start_time + segment.end_time) / 2.0

    def _assign_to_chapter(
        self,
        chapters: list[Chapter],
        transcript_segments: list[TranscriptSegment],
    ) -> dict[int, list[TranscriptSegment]]:
        chapter_bins: dict[int, list[TranscriptSegment]] = {i: [] for i in range(len(chapters))}
        for seg in transcript_segments:
            mid = self._segment_midpoint(seg)
            for idx, chapter in enumerate(chapters):
                if chapter.start_time <= mid < chapter.end_time:
                    chapter_bins[idx].append(seg)
                    break
            else:
                if chapters and mid >= chapters[-1].end_time:
                    chapter_bins[len(chapters) - 1].append(seg)
        return chapter_bins

    def _to_sentence_units(self, segments: list[TranscriptSegment]) -> list[SentenceUnit]:
        if not segments:
            return []

        units: list[SentenceUnit] = []
        cur_text_parts: list[str] = []
        cur_start = segments[0].start_time
        cur_end = segments[0].end_time

        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if not cur_text_parts:
                cur_start = seg.start_time
            cur_text_parts.append(text)
            cur_end = seg.end_time

            should_close = bool(self.SENTENCE_END_RE.search(text))
            if should_close:
                units.append(
                    SentenceUnit(
                        start_time=cur_start,
                        end_time=cur_end,
                        text=" ".join(cur_text_parts).strip(),
                    )
                )
                cur_text_parts = []

        if cur_text_parts:
            units.append(
                SentenceUnit(start_time=cur_start, end_time=cur_end, text=" ".join(cur_text_parts).strip())
            )
        return units

    def _split_sentences_into_chunks(
        self,
        semantic_video_id: str,
        chapter: Chapter,
        sentence_units: list[SentenceUnit],
        chunk_counter_start: int,
    ) -> tuple[list[Chunk], int]:
        chunks: list[Chunk] = []
        cur_group: list[SentenceUnit] = []
        chunk_counter = chunk_counter_start

        def flush_group() -> None:
            nonlocal chunk_counter, cur_group
            if not cur_group:
                return
            transcript = " ".join(item.text for item in cur_group).strip()
            chunk = Chunk(
                chunk_id=f"cs50_{semantic_video_id}_{chunk_counter:03d}",
                video_id=semantic_video_id,
                chapter_title=chapter.chapter_title,
                start_time=max(chapter.start_time, cur_group[0].start_time),
                end_time=min(chapter.end_time, cur_group[-1].end_time),
                transcript=transcript,
            )
            chunks.append(chunk)
            chunk_counter += 1
            cur_group = []

        for unit in sentence_units:
            if not cur_group:
                cur_group.append(unit)
                continue

            projected_end = unit.end_time
            projected_duration = projected_end - cur_group[0].start_time
            if projected_duration <= self.max_chunk_seconds:
                cur_group.append(unit)
                continue

            flush_group()
            cur_group.append(unit)

        flush_group()

        if not chunks:
            chunks.append(
                Chunk(
                    chunk_id=f"cs50_{semantic_video_id}_{chunk_counter:03d}",
                    video_id=semantic_video_id,
                    chapter_title=chapter.chapter_title,
                    start_time=chapter.start_time,
                    end_time=chapter.end_time,
                    transcript="",
                )
            )
            chunk_counter += 1

        return chunks, chunk_counter

    def build_chunks(
        self,
        semantic_video_id: str,
        chapters: list[Chapter],
        transcript_segments: list[TranscriptSegment],
    ) -> list[Chunk]:
        chapter_bins = self._assign_to_chapter(chapters, transcript_segments)

        all_chunks: list[Chunk] = []
        chunk_counter = 1
        for idx, chapter in enumerate(tqdm(chapters, desc="Semantic chunking by chapter")):
            segments = sorted(chapter_bins.get(idx, []), key=lambda s: s.start_time)
            sentence_units = self._to_sentence_units(segments)
            chapter_chunks, chunk_counter = self._split_sentences_into_chunks(
                semantic_video_id=semantic_video_id,
                chapter=chapter,
                sentence_units=sentence_units,
                chunk_counter_start=chunk_counter,
            )
            all_chunks.extend(chapter_chunks)

        return all_chunks


class FrameExtractor:
    """Extract one JPG frame at each chunk midpoint via ffmpeg."""

    def __init__(self, cookies_file: Path | None, cookies_from_browser: str | None) -> None:
        self._cookies_file = cookies_file
        self._cookies_from_browser = cookies_from_browser

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            try:
                import imageio_ffmpeg  # type: ignore

                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:
                raise RuntimeError(
                    "ffmpeg is not available in PATH and imageio-ffmpeg fallback could not be loaded."
                ) from exc

        self.ffmpeg_bin = ffmpeg_path

        self._ydl_opts: dict[str, Any] = apply_yt_auth_opts({
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": "best[ext=mp4]/best",
        }, cookies_file=cookies_file, cookies_from_browser=cookies_from_browser)

    def _run_ffmpeg_extract(self, input_source: str, midpoint: float, output_file: Path) -> None:
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rw_timeout",
            "15000000",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "5",
            "-ss",
            f"{midpoint:.3f}",
            "-i",
            input_source,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_file),
        ]
        subprocess.run(cmd, check=True, timeout=45)

    def _download_video_fallback(self, video_url: str, temp_dir: Path) -> Path:
        output_template = str(temp_dir / "source.%(ext)s")
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[ext=mp4]/best",
            "outtmpl": output_template,
        }
        opts = apply_yt_auth_opts(
            opts,
            cookies_file=self._cookies_file,
            cookies_from_browser=self._cookies_from_browser,
        )
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                downloaded = ydl.prepare_filename(info)
            video_path = Path(downloaded)
            if not video_path.exists():
                raise RuntimeError("yt-dlp reported success but local video file is missing.")
            return video_path
        except Exception as exc:
            raise RuntimeError(f"Failed to download local fallback video: {exc}") from exc

    def _resolve_stream_url(self, video_url: str) -> str:
        try:
            with yt_dlp.YoutubeDL(self._ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
            stream_url = info.get("url")
            if not stream_url:
                raise RuntimeError("yt-dlp did not return a direct stream URL.")
            return str(stream_url)
        except Exception as exc:
            raise RuntimeError(f"Failed to resolve direct stream URL: {exc}") from exc

    def extract_for_chunks(self, video_url: str, chunks: list[Chunk], frames_dir: Path) -> None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        stream_url = self._resolve_stream_url(video_url)
        local_video_path: Path | None = None

        with tempfile.TemporaryDirectory(prefix="cs50_ingest_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            for chunk in tqdm(chunks, desc="Extracting midpoint frames"):
                midpoint = (chunk.start_time + chunk.end_time) / 2.0
                output_file = frames_dir / f"{chunk.chunk_id}.jpg"
                last_error: Exception | None = None

                # If fallback local video already exists, extract directly from local file.
                if local_video_path is not None:
                    try:
                        self._run_ffmpeg_extract(str(local_video_path), midpoint, output_file)
                        continue
                    except subprocess.CalledProcessError as exc:
                        raise RuntimeError(
                            f"Local fallback extraction failed for {chunk.chunk_id} at {midpoint:.3f}s: {exc}"
                        ) from exc

                for attempt in range(3):
                    if attempt > 0:
                        stream_url = self._resolve_stream_url(video_url)

                    try:
                        self._run_ffmpeg_extract(stream_url, midpoint, output_file)
                        last_error = None
                        break
                    except subprocess.CalledProcessError as exc:
                        last_error = exc

                if last_error is None:
                    continue

                # Network stream failed after retries: download a temporary local copy once,
                # then continue extraction from local file for this and subsequent chunks.
                local_video_path = self._download_video_fallback(video_url=video_url, temp_dir=temp_dir)
                try:
                    self._run_ffmpeg_extract(str(local_video_path), midpoint, output_file)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"Frame extraction failed for {chunk.chunk_id} at {midpoint:.3f}s from both stream and local file: {exc}"
                    ) from exc


class SemanticIngestionPipeline:
    def __init__(
        self,
        chunk_seconds: float,
        metadata_path: Path | None,
        cookies_file: Path | None,
        cookies_from_browser: str | None,
    ) -> None:
        self.cookies_file = cookies_file
        self.cookies_from_browser = cookies_from_browser
        self.chapter_extractor = ChapterExtractor(
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
        )
        self.transcript_fetcher = TranscriptFetcher()
        self.chunker = SemanticChunker(max_chunk_seconds=chunk_seconds)
        self.metadata_resolver = MetadataResolver(metadata_path=metadata_path)

    @staticmethod
    def extract_video_id(url: str) -> str:
        patterns = [
            re.compile(r"(?:v=)([A-Za-z0-9_-]{11})"),
            re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
            re.compile(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})"),
        ]
        for pattern in patterns:
            match = pattern.search(url)
            if match:
                return match.group(1)
        raise ValueError("Could not parse YouTube video ID from URL.")

    @staticmethod
    def save_chunks(chunks: list[Chunk], output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "video_id": chunk.video_id,
                "chapter_title": chunk.chapter_title,
                "start_time": round(chunk.start_time, 3),
                "end_time": round(chunk.end_time, 3),
                "transcript": chunk.transcript,
            }
            for chunk in chunks
        ]
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def run(self, url: str, chunks_output: Path, frames_dir: Path) -> None:
        youtube_video_id = self.extract_video_id(url)
        semantic_video_id = self.metadata_resolver.resolve_semantic_id(youtube_video_id)

        info = self.chapter_extractor.fetch_info(url)
        chapters = self.chapter_extractor.extract_chapters(info)

        transcript_segments = self.transcript_fetcher.fetch(youtube_video_id)
        chunks = self.chunker.build_chunks(
            semantic_video_id=semantic_video_id,
            chapters=chapters,
            transcript_segments=transcript_segments,
        )

        self.save_chunks(chunks, chunks_output)

        frame_extractor = FrameExtractor(
            cookies_file=self.cookies_file,
            cookies_from_browser=self.cookies_from_browser,
        )
        frame_extractor.extract_for_chunks(video_url=url, chunks=chunks, frames_dir=frames_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic Ingestion & Chunking for CS50x QA video pipeline.")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=60.0,
        help="Maximum sub-chunk duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        default=Path("data/interim/text_chunks/chunks.json"),
        help="Output path for chunks.json",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/interim/frames"),
        help="Output directory for extracted JPG frames",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/playlist_metadata.json"),
        help="Path to playlist metadata JSON for semantic video_id naming",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        default=None,
        help="Path to Netscape format cookies file for yt-dlp YouTube authentication",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Browser name for yt-dlp cookie extraction (e.g., chrome, edge, firefox)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        pipeline = SemanticIngestionPipeline(
            chunk_seconds=args.chunk_seconds,
            metadata_path=args.metadata,
            cookies_file=args.cookies_file,
            cookies_from_browser=args.cookies_from_browser,
        )
        pipeline.run(url=args.url, chunks_output=args.chunks_output, frames_dir=args.frames_dir)
        print(f"Done. Chunks saved to: {args.chunks_output}")
        print(f"Frames saved to: {args.frames_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
