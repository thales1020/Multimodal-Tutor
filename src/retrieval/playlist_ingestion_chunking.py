from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass
class PlaylistVideo:
    url: str
    video_id: str
    week: int
    semantic_video_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ingestion_chunking.py for each video in playlist_metadata.json"
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/playlist_metadata.json"),
        help="Path to playlist metadata JSON",
    )
    parser.add_argument(
        "--text-chunks-dir",
        type=Path,
        default=Path("data/interim/text_chunks"),
        help="Directory for per-video chunk files",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/interim/frames"),
        help="Directory for extracted frames",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=60.0,
        help="Maximum chunk duration in seconds",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run videos even if output chunk file already exists",
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


def load_playlist(metadata_path: Path) -> list[PlaylistVideo]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("playlist_metadata.json must be a list of objects")

    videos: list[PlaylistVideo] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        url = str(row.get("url") or "").strip()
        video_id = str(row.get("video_id") or "").strip()
        semantic_video_id = str(row.get("semantic_video_id") or "").strip()
        week = int(row.get("week") or 0)

        if not url or not video_id or not semantic_video_id:
            continue

        videos.append(
            PlaylistVideo(
                url=url,
                video_id=video_id,
                week=week,
                semantic_video_id=semantic_video_id,
            )
        )

    return videos


def build_child_command(
    ingestion_script_path: Path,
    video: PlaylistVideo,
    metadata_path: Path,
    chunks_output: Path,
    frames_dir: Path,
    chunk_seconds: float,
    cookies_file: Path | None,
    cookies_from_browser: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ingestion_script_path),
        "--url",
        video.url,
        "--chunk-seconds",
        str(chunk_seconds),
        "--chunks-output",
        str(chunks_output),
        "--frames-dir",
        str(frames_dir),
        "--metadata",
        str(metadata_path),
    ]
    if cookies_file is not None:
        cmd.extend(["--cookies-file", str(cookies_file)])
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    return cmd


def run_single_video(
    ingestion_script_path: Path,
    video: PlaylistVideo,
    metadata_path: Path,
    text_chunks_dir: Path,
    frames_dir: Path,
    chunk_seconds: float,
    cookies_file: Path | None,
    cookies_from_browser: str | None,
) -> None:
    chunks_output = text_chunks_dir / f"{video.semantic_video_id}_chunks.json"
    cmd = build_child_command(
        ingestion_script_path=ingestion_script_path,
        video=video,
        metadata_path=metadata_path,
        chunks_output=chunks_output,
        frames_dir=frames_dir,
        chunk_seconds=chunk_seconds,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr if stderr else stdout
        raise RuntimeError(
            f"ingestion_chunking failed for week={video.week}, semantic_video_id={video.semantic_video_id}. {detail}"
        )


def main() -> int:
    args = parse_args()

    text_chunks_dir = args.text_chunks_dir
    frames_dir = args.frames_dir
    text_chunks_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    videos = load_playlist(args.metadata_path)
    ingestion_script_path = Path(__file__).with_name("ingestion_chunking.py")
    if not ingestion_script_path.exists():
        raise FileNotFoundError(f"Missing child script: {ingestion_script_path}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for video in tqdm(videos, desc="Playlist Ingestion", unit="video"):
        chunks_output = text_chunks_dir / f"{video.semantic_video_id}_chunks.json"
        if chunks_output.exists() and not args.overwrite:
            skip_count += 1
            continue

        try:
            run_single_video(
                ingestion_script_path=ingestion_script_path,
                video=video,
                metadata_path=args.metadata_path,
                text_chunks_dir=text_chunks_dir,
                frames_dir=frames_dir,
                chunk_seconds=args.chunk_seconds,
                cookies_file=args.cookies_file,
                cookies_from_browser=args.cookies_from_browser,
            )
            success_count += 1
        except Exception as exc:
            fail_count += 1
            print(f"[ERROR] {exc}")

    print(
        f"Done. total={len(videos)} success={success_count} skipped={skip_count} failed={fail_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
