"""
Multimodal Extraction & Cleaning Pipeline for CS50x video frames.

This module processes chunks with corresponding frames using a local
OpenAI-compatible vision model to extract OCR text and visual descriptions,
with robust batching and retries.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a video chunk with extracted multimodal data."""

    chunk_id: str
    video_id: str
    chapter_title: str
    start_time: float
    end_time: float
    transcript: str
    ocr_text: str = ""
    visual_description: str = ""


@dataclass
class VLMResponse:
    """Parsed response from vision-language model API."""

    is_valid: bool
    ocr_text: str
    visual_description: str

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> VLMResponse:
        if not isinstance(data, dict):
            raise ValueError("Response must be a dict")

        is_valid = bool(data.get("is_valid", False))
        ocr_text = str(data.get("ocr_text", "")).strip()
        visual_description = str(data.get("visual_description", "")).strip()
        return cls(is_valid=is_valid, ocr_text=ocr_text, visual_description=visual_description)


class ChunkLoader:
    @staticmethod
    def load_chunks(chunks_path: Path) -> list[Chunk]:
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        with chunks_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Chunks JSON must be a list of dicts")

        chunks: list[Chunk] = []
        for item in data:
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict chunk entry: %s", item)
                continue

            chunks.append(
                Chunk(
                    chunk_id=str(item.get("chunk_id", "")),
                    video_id=str(item.get("video_id", "")),
                    chapter_title=str(item.get("chapter_title", "")),
                    start_time=float(item.get("start_time", 0.0)),
                    end_time=float(item.get("end_time", 0.0)),
                    transcript=str(item.get("transcript", "")),
                    ocr_text=str(item.get("ocr_text", "")),
                    visual_description=str(item.get("visual_description", "")),
                )
            )

        return chunks


class FrameFinder:
    def __init__(self, frames_dir: Path) -> None:
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        self.frames_dir = frames_dir

    def find_frame(self, chunk_id: str) -> Optional[Path]:
        frame_path = self.frames_dir / f"{chunk_id}.jpg"
        return frame_path if frame_path.exists() else None


class OpenAIExtractor:
    """Calls a local OpenAI-compatible vision model for multimodal extraction."""

    def __init__(
        self,
        api_key: str,
        system_prompt_path: Path,
        request_delay_seconds: float = 2.0,
        model_name: str = "llama3.2-vision",
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        self.client = OpenAI(api_key=api_key or "ollama", base_url=base_url)
        self.model_name = model_name
        self.base_url = base_url
        self.request_delay_seconds = max(0.0, request_delay_seconds)
        self._request_lock = asyncio.Lock()
        self._next_request_time = 0.0

        if not system_prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
        with system_prompt_path.open("r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

        logger.info(
            "OpenAIExtractor initialized with model=%s, base_url=%s, request_delay_seconds=%.2f",
            self.model_name,
            self.base_url,
            self.request_delay_seconds,
        )

    async def _throttle_before_request(self) -> None:
        async with self._request_lock:
            now = time.monotonic()
            wait_seconds = self._next_request_time - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._next_request_time = time.monotonic() + self.request_delay_seconds

    @staticmethod
    def _extract_retry_delay_seconds(exc: Exception) -> Optional[float]:
        msg = str(exc)
        retry_match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", msg, flags=re.IGNORECASE)
        if retry_match:
            return float(retry_match.group(1))
        return None

    @staticmethod
    def _extract_json_text(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)

        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return stripped[first_brace : last_brace + 1]

        return stripped

    @staticmethod
    def _image_to_base64(image_path: Path) -> str:
        with image_path.open("rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _build_multi_chunk_prompt(chunk_ids: list[str]) -> str:
        chunk_list = ", ".join(chunk_ids)
        return (
            "You will receive multiple images in this request. "
            "Each image corresponds to one chunk_id listed below, in exact order.\n"
            f"chunk_ids_order: [{chunk_list}]\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            '  "results": [\n'
            "    {\n"
            '      "chunk_id": "<chunk id>",\n'
            '      "is_valid": true or false,\n'
            '      "ocr_text": "...",\n'
            '      "visual_description": "..."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Keep results in the same order as chunk_ids_order.\n"
            "- Include exactly one result object per chunk_id.\n"
            "- Do not include markdown or any non-JSON text."
        )

    async def extract_batch_async(self, items: list[tuple[str, Path]]) -> dict[str, VLMResponse]:
        if not items:
            return {}

        chunk_ids = [chunk_id for chunk_id, _ in items]
        batch_prompt = self._build_multi_chunk_prompt(chunk_ids)

        try:
            attempt_num = 0
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((Exception,)),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
            ):
                with attempt:
                    attempt_num += 1
                    try:
                        await self._throttle_before_request()

                        user_content: list[dict[str, Any]] = [{"type": "text", "text": batch_prompt}]
                        for idx, (chunk_id, image_path) in enumerate(items, start=1):
                            image_b64 = self._image_to_base64(image_path)
                            user_content.append(
                                {"type": "text", "text": f"Image {idx} chunk_id: {chunk_id}"}
                            )
                            user_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                                }
                            )

                        response = await asyncio.to_thread(
                            self.client.chat.completions.create,
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": user_content},
                            ],
                            temperature=0,
                        )

                        content = response.choices[0].message.content if response.choices else None
                        if not content:
                            raise ValueError("Empty response from OpenAI API")

                        json_text = self._extract_json_text(content)
                        parsed = json.loads(json_text)
                        if not isinstance(parsed, dict) or "results" not in parsed:
                            raise ValueError(f"Batch response missing results field: {parsed}")

                        raw_results = parsed.get("results", [])
                        if not isinstance(raw_results, list):
                            raise ValueError(f"Batch response results must be list: {parsed}")

                        result_map: dict[str, VLMResponse] = {}
                        for raw_item in raw_results:
                            if not isinstance(raw_item, dict):
                                continue
                            raw_chunk_id = str(raw_item.get("chunk_id", "")).strip()
                            if not raw_chunk_id:
                                continue
                            result_map[raw_chunk_id] = VLMResponse.from_json(raw_item)

                        missing_ids = [cid for cid in chunk_ids if cid not in result_map]
                        if missing_ids:
                            raise ValueError(
                                f"Batch response missing chunk_ids: {missing_ids}. Full response: {parsed}"
                            )

                        logger.info(
                            "Successfully extracted batch of %d chunks on attempt %d",
                            len(items),
                            attempt_num,
                        )
                        return result_map

                    except Exception as api_exc:
                        if "429" in str(api_exc):
                            retry_after = self._extract_retry_delay_seconds(api_exc)
                            if retry_after is not None:
                                sleep_seconds = min(retry_after, 120.0)
                                logger.warning(
                                    "Batch with first chunk %s hit rate limit, waiting %.1fs before retry",
                                    chunk_ids[0],
                                    sleep_seconds,
                                )
                                await asyncio.sleep(sleep_seconds)

                        logger.warning(
                            "Attempt %d/%d for batch starting at chunk %s failed: %s. Retrying...",
                            attempt_num,
                            3,
                            chunk_ids[0],
                            api_exc,
                        )
                        raise

        except RetryError as exc:
            logger.error(
                "Retries exhausted for batch starting at chunk %s after %d attempts. Last error: %s",
                chunk_ids[0],
                attempt_num,
                exc.last_attempt.exception() if exc.last_attempt else "Unknown",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to extract batch starting at chunk {chunk_ids[0]} after {attempt_num} retries"
            ) from exc


class MultimodalExtractionPipeline:
    def __init__(
        self,
        chunks_path: Path,
        frames_dir: Path,
        api_key: str,
        system_prompt_path: Path,
        output_path: Path,
        model_name: str = "gpt-4o",
        request_delay_seconds: float = 2.0,
        continue_on_error: bool = False,
        strict_retry_wait_seconds: float = 30.0,
        batch_size: int = 20,
        batch_pause_seconds: float = 30.0,
        chunks_per_request: int = 5,
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        self.chunks_path = chunks_path
        self.frames_dir = frames_dir
        self.output_path = output_path
        self.continue_on_error = continue_on_error
        self.strict_retry_wait_seconds = max(1.0, strict_retry_wait_seconds)
        self.batch_size = max(1, batch_size)
        self.batch_pause_seconds = max(0.0, batch_pause_seconds)
        self.chunks_per_request = chunks_per_request

        self.chunk_loader = ChunkLoader()
        self.frame_finder = FrameFinder(frames_dir)
        self.extractor = OpenAIExtractor(
            api_key=api_key,
            system_prompt_path=system_prompt_path,
            request_delay_seconds=request_delay_seconds,
            model_name=model_name,
            base_url=base_url,
        )

    async def run_async(self) -> list[Chunk]:
        logger.info("Starting multimodal extraction pipeline")

        chunks = self.chunk_loader.load_chunks(self.chunks_path)
        logger.info("Loaded %d chunks from %s", len(chunks), self.chunks_path)

        processed_chunks: list[Chunk] = []
        failed_chunks = 0

        with tqdm(total=len(chunks), desc="Extracting multimodal data (OpenAI)") as pbar:
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
            for batch_index in range(total_batches):
                start = batch_index * self.batch_size
                end = min(start + self.batch_size, len(chunks))
                batch_chunks = chunks[start:end]
                logger.info(
                    "Processing batch %d/%d (chunks %d-%d)",
                    batch_index + 1,
                    total_batches,
                    start + 1,
                    end,
                )

                request_groups: list[list[Chunk]] = []
                for i in range(0, len(batch_chunks), self.chunks_per_request):
                    request_groups.append(batch_chunks[i : i + self.chunks_per_request])

                for group in request_groups:
                    missing_frame_chunks: list[Chunk] = []
                    request_items: list[tuple[str, Path]] = []

                    for chunk in group:
                        frame_path = self.frame_finder.find_frame(chunk.chunk_id)
                        if frame_path is None:
                            missing_frame_chunks.append(chunk)
                        else:
                            request_items.append((chunk.chunk_id, frame_path))

                    for chunk in missing_frame_chunks:
                        logger.warning("Frame not found for chunk %s, skipping extraction", chunk.chunk_id)
                        processed_chunks.append(chunk)
                        pbar.update(1)

                    if not request_items:
                        continue

                    while True:
                        try:
                            batch_results = await self.extractor.extract_batch_async(request_items)
                            for chunk in group:
                                if chunk.chunk_id not in batch_results:
                                    continue
                                vlm_response = batch_results[chunk.chunk_id]
                                if not vlm_response.is_valid:
                                    chunk.ocr_text = ""
                                    chunk.visual_description = ""
                                else:
                                    chunk.ocr_text = vlm_response.ocr_text
                                    chunk.visual_description = vlm_response.visual_description
                                processed_chunks.append(chunk)
                                pbar.update(1)
                            break

                        except Exception as exc:
                            failed_chunks += len(request_items)
                            first_chunk_id = request_items[0][0]
                            logger.error(
                                "Error processing request group starting at chunk %s: %s",
                                first_chunk_id,
                                exc,
                                exc_info=True,
                            )

                            if self.continue_on_error:
                                for chunk_id, _ in request_items:
                                    chunk_obj = next((c for c in group if c.chunk_id == chunk_id), None)
                                    if chunk_obj is None:
                                        continue
                                    chunk_obj.ocr_text = ""
                                    chunk_obj.visual_description = ""
                                    processed_chunks.append(chunk_obj)
                                    pbar.update(1)
                                break

                            logger.warning(
                                "Request group starting at chunk %s failed. Retrying same group in %.1fs.",
                                first_chunk_id,
                                self.strict_retry_wait_seconds,
                            )
                            await asyncio.sleep(self.strict_retry_wait_seconds)

                is_last_batch = batch_index == total_batches - 1
                if not is_last_batch and self.batch_pause_seconds > 0:
                    logger.info(
                        "Batch %d/%d completed. Pausing %.1fs before next batch.",
                        batch_index + 1,
                        total_batches,
                        self.batch_pause_seconds,
                    )
                    await asyncio.sleep(self.batch_pause_seconds)

        if failed_chunks:
            logger.warning("Pipeline encountered %d failed attempts during strict retries", failed_chunks)

        return processed_chunks

    def run(self) -> list[Chunk]:
        return asyncio.run(self.run_async())

    def save_results(self, chunks: list[Chunk]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(chunk) for chunk in chunks]
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d processed chunks to %s", len(chunks), self.output_path)


def parse_args() -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Multimodal Extraction & Cleaning Pipeline for CS50x video frames (local OpenAI-compatible vision model)."
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/interim/text_chunks"),
        help="Path to one chunk JSON file or a directory containing chunk JSON files",
    )
    parser.add_argument(
        "--chunks-glob",
        type=str,
        default="week*_chunks.json",
        help="Glob used when --chunks-path points to a directory",
    )
    parser.add_argument("--frames-dir", type=Path, default=Path("data/interim/frames"))
    parser.add_argument("--api-key", type=str, help="OpenAI API key (overrides .env file)")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--system-prompt-path", type=Path, default=Path("data/system_prompt.txt"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path; defaults to a filename derived from --chunks-path",
    )
    parser.add_argument("--model", type=str, default="llama3.2-vision", help="Local OpenAI-compatible vision model name")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="OpenAI-compatible local endpoint base URL (default: Ollama)",
    )
    parser.add_argument("--request-delay", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--strict-retry-wait", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--batch-pause", type=float, default=30.0)
    parser.add_argument(
        "--chunks-per-request",
        type=int,
        default=5,
        choices=[1, 2, 5, 8, 10],
        help="Number of frames grouped into one local vision-model request",
    )
    return vars(parser.parse_args())


def derive_output_path(chunks_path: Path) -> Path:
    return Path("data/processed") / chunks_path.name


def resolve_chunk_paths(chunks_path: Path, chunks_glob: str) -> list[Path]:
    if chunks_path.is_file():
        return [chunks_path]

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks path not found: {chunks_path}")

    if not chunks_path.is_dir():
        raise ValueError(f"Chunks path must be a file or directory: {chunks_path}")

    candidates = sorted(p for p in chunks_path.glob(chunks_glob) if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No chunk files matched '{chunks_glob}' under {chunks_path}"
        )

    return candidates


def load_api_key(api_key_arg: Optional[str], env_file: Path) -> str:
    if api_key_arg:
        return api_key_arg

    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    from dotenv import dotenv_values

    env_vars = dotenv_values(env_file)
    api_key = (
        str(env_vars.get("OPENAI_API_KEY") or "").strip().strip('"\'')
        or str(env_vars.get("API_KEY") or "").strip().strip('"\'')
    )

    if not api_key:
        logger.info("No API key found; using dummy key for local OpenAI-compatible endpoint")
        return "ollama"

    return api_key


def main() -> int:
    try:
        args = parse_args()
        api_key = load_api_key(args["api_key"], args["env_file"])
        chunk_paths = resolve_chunk_paths(args["chunks_path"], args["chunks_glob"])

        if args["output_path"] and len(chunk_paths) > 1:
            if args["output_path"].suffix:
                raise ValueError(
                    "When processing multiple chunk files, --output-path must be a directory"
                )
            output_dir = args["output_path"]
        elif args["output_path"] and len(chunk_paths) == 1:
            output_dir = None
        else:
            output_dir = None

        for chunk_path in chunk_paths:
            output_path = args["output_path"]
            if output_dir is not None:
                output_path = output_dir / chunk_path.name
            elif output_path is None:
                output_path = derive_output_path(chunk_path)

            logger.info("Processing chunk file: %s", chunk_path)
            logger.info("Output file: %s", output_path)

            pipeline = MultimodalExtractionPipeline(
                chunks_path=chunk_path,
                frames_dir=args["frames_dir"],
                api_key=api_key,
                system_prompt_path=args["system_prompt_path"],
                output_path=output_path,
                model_name=args["model"],
                base_url=args["base_url"],
                request_delay_seconds=args["request_delay"],
                continue_on_error=args["continue_on_error"],
                strict_retry_wait_seconds=args["strict_retry_wait"],
                batch_size=args["batch_size"],
                batch_pause_seconds=args["batch_pause"],
                chunks_per_request=args["chunks_per_request"],
            )

            processed_chunks = pipeline.run()
            pipeline.save_results(processed_chunks)

        logger.info("Pipeline completed successfully")
        return 0

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
