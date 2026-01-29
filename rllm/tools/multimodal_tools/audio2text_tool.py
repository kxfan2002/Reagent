import asyncio
import math
import os
import random
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Tuple

import aiohttp

from rllm.tools.tool_base import Tool, ToolOutput

# Default configuration
DEFAULT_TIMEOUT = 60
WHISPER_API_URL = os.getenv("WHISPER_API", "http://127.0.0.1:8000/v1/audio/transcriptions")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper")
MAX_DURATION = int(os.getenv("MAX_DURATION", "30"))
AUDIO_BASE_PATH = os.getenv("AUDIO_BASE_PATH", "./data")
TEMP_DIR = os.getenv("AUDIO_TEMP_DIR", "./tmp/audio_temp")
MAX_AUDIO_MB = int(os.getenv("AUDIO_MAX_SIZE_MB", "100"))

# Supported audio formats
SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".aac", ".wma", ".opus"
}

# Create temp directory
os.makedirs(TEMP_DIR, exist_ok=True)


def _validate_audio_file(audio_path: str, max_size_mb: int) -> Tuple[bool, str]:
    """Validate audio file existence, extension, and size."""
    try:
        if not os.path.exists(audio_path):
            return False, f"Audio file does not exist: {audio_path}"

        file_info = Path(audio_path)
        ext = file_info.suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            return False, f"Unsupported audio extension: {ext}. Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"

        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return False, "Audio file is empty (0 bytes)."

        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"Audio file is too large ({file_size_mb:.2f} MB). Limit: {max_size_mb} MB."

    except Exception as exc:
        return False, f"Audio validation failed: {str(exc)}"

    return True, ""


def get_audio_duration(audio_path: str) -> float:
    """
    Robust audio duration detection without requiring ffmpeg/ffprobe.
    Priority:
        1) mutagen (accurate for MP3 / M4A / OGG / FLAC / etc.)
        2) wave (for WAV only)
        3) file size fallback (unreliable)
    """
    if not os.path.exists(audio_path):
        return 0.0

    # --- Try mutagen (recommended) ---
    try:
        from mutagen import File as MutagenFile
        mf = MutagenFile(audio_path)
        if mf is not None:
            dur = getattr(mf.info, "length", None)
            if dur and dur > 0:
                return float(dur)
    except ImportError:
        print("[Audio2Text] Warning: mutagen not installed. Install with 'pip install mutagen' for accurate duration detection.")
    except Exception as e:
        print(f"[Audio2Text] mutagen failed: {e}")

    # --- Try Python wave module for WAV only ---
    try:
        import wave
        if audio_path.lower().endswith(".wav"):
            with wave.open(audio_path, "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                return frames / float(rate)
    except Exception:
        pass

    # --- Fallback: estimate from file size (not reliable) ---
    try:
        size = os.path.getsize(audio_path)
        est = size * 8 / (128 * 1000)  # assume 128 kbps
        # print(f"[Audio2Text] Warning: Using file size estimation for duration ({est:.1f}s). May be inaccurate.")
        return est
    except Exception:
        pass

    return 0.0


def split_single_segment(idx: int, audio_path: str, segment_duration: int, temp_dir: str) -> Optional[str]:
    """
    Split a single audio segment using ffmpeg.
    
    Args:
        idx: Segment index
        audio_path: Original audio file path
        segment_duration: Duration of each segment in seconds
        temp_dir: Directory to store segments
    
    Returns:
        Path to the segment file, or None if failed
    """
    start_time = idx * segment_duration
    segment_path = os.path.join(temp_dir, f"segment_{idx:03d}.wav")
    
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-ss", str(start_time),
        "-t", str(segment_duration),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        "-y",
        segment_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(segment_path):
            return segment_path
        else:
            print(f"[Audio2Text] FFmpeg segment {idx} failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"[Audio2Text] FFmpeg segment {idx} timed out")
        return None
    except Exception as e:
        print(f"[Audio2Text] FFmpeg segment {idx} error: {e}")
        return None


def split_audio_multithread(audio_path: str, segment_duration: int, temp_dir: str) -> List[str]:
    """
    Split audio into segments using multi-threading.
    
    Args:
        audio_path: Path to the audio file
        segment_duration: Duration of each segment in seconds
        temp_dir: Directory to store segments
    
    Returns:
        List of segment file paths
    """
    duration = get_audio_duration(audio_path)
    
    # If audio is short enough, don't split
    if duration <= segment_duration:
        return [audio_path]

    num_segments = math.ceil(duration / segment_duration)
    segment_temp_dir = tempfile.mkdtemp(prefix="audio_split_", dir=temp_dir)
    segments: List[str] = []

    # Use ThreadPoolExecutor for parallel splitting
    with ThreadPoolExecutor(max_workers=min(8, num_segments)) as executor:
        futures = {
            executor.submit(split_single_segment, i, audio_path, segment_duration, segment_temp_dir): i
            for i in range(num_segments)
        }
        
        for future in as_completed(futures):
            seg_path = future.result()
            if seg_path:
                segments.append(seg_path)

    # If splitting failed, return original file
    if not segments:
        print("[Audio2Text] Audio splitting failed, using original file")
        return [audio_path]

    # Sort segments by index
    segments.sort()
    return segments


def _extract_text_from_response(response: Any) -> Tuple[bool, str]:
    """Extract transcription text from API response."""
    if isinstance(response, dict):
        if response.get("success"):
            text = response.get("text", "")
            if text:
                return True, text
            return False, "Empty transcription result."
        return False, response.get("error", "Unknown transcription error.")
    return False, "Unexpected response format from transcription service."


class Audio2TextTool(Tool):
    """A tool for converting audio files into text using Whisper API."""

    NAME = "audio2text"
    DESCRIPTION = "Convert audio file into text using automatic speech recognition (ASR)."

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        api_url: str = WHISPER_API_URL,
        model: str = WHISPER_MODEL,
        max_duration: int = MAX_DURATION,
        max_audio_size_mb: int = MAX_AUDIO_MB,
        timeout: float = DEFAULT_TIMEOUT,
        max_retry: int = 3,
        max_output_length: int = 10000,
        temp_dir: str = TEMP_DIR,
    ):
        """
        Initialize the Audio2Text tool.

        Args:
            name (str): The name of the tool.
            description (str): A description of the tool's purpose.
            api_url (str): Whisper API endpoint URL.
            model (str): Model name to use for transcription.
            max_duration (int): Maximum duration per segment in seconds.
            max_audio_size_mb (int): Maximum allowed audio file size in MB.
            timeout (float): Maximum time in seconds to wait for API response.
            max_retry (int): Maximum number of retry attempts.
            max_output_length (int): Maximum length of output text before truncation.
            temp_dir (str): Directory for temporary files.
        """
        self.api_url = api_url
        self.model = model
        self.max_duration = max_duration
        self.max_audio_size_mb = max_audio_size_mb
        self.timeout = timeout
        self.max_retry = max_retry
        self.max_output_length = max_output_length
        self.temp_dir = temp_dir
        super().__init__(name=name, description=description)

    @property
    def json(self):
        """Return the tool's information in the required format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Convert audio into text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "The local file path or URL of the audio file to be transcribed."
                        }
                    },
                    "required": ["audio_path"]
                }
            }
        }

    async def _transcribe_single_file(self, session: aiohttp.ClientSession, audio_path: str) -> dict[str, Any]:
        """
        Transcribe a single audio file using Whisper API.
        
        Args:
            session: aiohttp ClientSession
            audio_path: Path to the audio file
        
        Returns:
            Dict containing success status and transcription text or error
        """
        for attempt in range(self.max_retry):
            try:
                form = aiohttp.FormData()
                form.add_field("model", self.model)

                with open(audio_path, "rb") as f:
                    form.add_field(
                        "file",
                        f,
                        filename=os.path.basename(audio_path),
                        content_type="audio/wav"
                    )

                    async with session.post(self.api_url, data=form, timeout=self.timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return {
                                "success": True,
                                "text": data.get("text", ""),
                                "audio_path": audio_path
                            }
                        elif resp.status == 429:
                            # Rate limit, wait and retry
                            wait_time = 2.0 * (2 ** attempt)
                            # print(f"[Audio2Text] Rate limited (429), waiting {wait_time:.1f}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            error_text = await resp.text()
                            return {
                                "success": False,
                                "error": f"HTTP {resp.status}: {error_text}"
                            }

            except asyncio.TimeoutError:
                if attempt == self.max_retry - 1:
                    return {"success": False, "error": "Request timed out after retries"}
                print(f"[Audio2Text] Timeout on attempt {attempt + 1}/{self.max_retry}, retrying...")
                await asyncio.sleep(1.0 * (2 ** attempt))
            except Exception as e:
                if attempt == self.max_retry - 1:
                    return {"success": False, "error": f"Transcription failed: {str(e)}"}
                print(f"[Audio2Text] Error on attempt {attempt + 1}/{self.max_retry}: {e}")
                await asyncio.sleep(1.0 * (2 ** attempt))

        return {"success": False, "error": "Max retries exceeded"}

    async def _async_transcribe(self, audio_path: str) -> ToolOutput:
        """
        Internal async method for transcribing audio.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            ToolOutput containing transcription or error
        """
        # Handle relative vs absolute paths
        full_audio_path = audio_path
        if AUDIO_BASE_PATH and not os.path.isabs(audio_path):
            full_audio_path = os.path.join(AUDIO_BASE_PATH, audio_path)

        # print(f"[Audio2Text] Full audio path: {full_audio_path}")
        # Validate audio file
        is_valid, validation_error = _validate_audio_file(full_audio_path, self.max_audio_size_mb)
        if not is_valid:
            return ToolOutput(name=self.name or "audio2text", error=validation_error)

        # Get audio duration
        duration = get_audio_duration(full_audio_path)
        # print(f"[Audio2Text] Audio duration: {duration:.1f}s, max per segment: {self.max_duration}s")

        # Decide whether to split
        if duration <= self.max_duration:
            # Short audio, transcribe directly
            async with aiohttp.ClientSession() as session:
                result = await self._transcribe_single_file(session, full_audio_path)
            
            success, payload = _extract_text_from_response(result)
            if success:
                # Truncate if needed
                if len(payload) > self.max_output_length:
                    payload = (
                        payload[:self.max_output_length]
                        + f"\n... (output truncated, total length: {len(payload)} chars)"
                    )
                return ToolOutput(name=self.name or "audio2text", output=payload)
            else:
                return ToolOutput(name=self.name or "audio2text", error=payload)

        # Long audio, need to split
        # print(f"[Audio2Text] Splitting audio into segments...")
        segments = split_audio_multithread(full_audio_path, self.max_duration, self.temp_dir)

        # If splitting failed, try original file
        if len(segments) == 1 and segments[0] == full_audio_path:
            async with aiohttp.ClientSession() as session:
                result = await self._transcribe_single_file(session, full_audio_path)
            
            success, payload = _extract_text_from_response(result)
            if success:
                if len(payload) > self.max_output_length:
                    payload = (
                        payload[:self.max_output_length]
                        + f"\n... (output truncated, total length: {len(payload)} chars)"
                    )
                return ToolOutput(name=self.name or "audio2text", output=payload)
            else:
                return ToolOutput(name=self.name or "audio2text", error=payload)

        # Parallel transcription of segments
        # print(f"[Audio2Text] Transcribing {len(segments)} segments in parallel...")
        async with aiohttp.ClientSession() as session:
            tasks = [self._transcribe_single_file(session, seg) for seg in segments]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect and merge results
        text_list = []
        errors = []
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Segment {idx}: {str(result)}")
                continue
            
            success, payload = _extract_text_from_response(result)
            if success:
                text_list.append((idx, payload))
            else:
                errors.append(f"Segment {idx}: {payload}")

        # Sort by index and join
        text_list.sort(key=lambda x: x[0])
        final_text = " ".join(t for _, t in text_list)

        if text_list:
            # Truncate if needed
            if len(final_text) > self.max_output_length:
                final_text = (
                    final_text[:self.max_output_length]
                    + f"\n... (output truncated, total length: {len(final_text)} chars)"
                )
            return ToolOutput(name=self.name or "audio2text", output=final_text)
        else:
            # All segments failed
            error_summary = "; ".join(errors[:3]) if errors else "Unknown transcription failure"
            return ToolOutput(name=self.name or "audio2text", error=f"Transcription failed: {error_summary}")

    def forward(self, audio_path: str) -> ToolOutput:
        """
        Synchronous transcription (wraps async implementation).
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            ToolOutput containing transcription or error
        """
        try:
            # Try to run in new event loop
            result = asyncio.run(self._async_transcribe(audio_path))
            return result
        except RuntimeError as exc:
            # Handle case where event loop is already running (e.g., in Jupyter)
            if "event loop is running" in str(exc).lower():
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self._async_transcribe(audio_path))
                    return result
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()
            else:
                return ToolOutput(
                    name=self.name or "audio2text",
                    error=f"Event loop error: {str(exc)}"
                )
        except Exception as e:
            return ToolOutput(
                name=self.name or "audio2text",
                error=f"Transcription error: {str(e)}"
            )

    async def async_forward(self, audio_path: str) -> ToolOutput:
        """
        Asynchronous transcription (native async implementation).
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            ToolOutput containing transcription or error
        """
        return await self._async_transcribe(audio_path)


if __name__ == "__main__":
    # Test the tool
    tool = Audio2TextTool()
    print("Testing Audio2Text Tool...")
    print("=" * 80)
    
    # Test with a sample audio file (replace with actual path)
    test_audio = "rl/audios/LongAudio/2.wav"
    
    # if os.path.exists(test_audio):
    print(f"\nTranscribing: {test_audio}")
    result = tool(audio_path=test_audio)
    print(result)
    # else:
    #     print(f"⚠️  Test audio not found: {test_audio}")
    #     print("Please set a valid audio path to test")
    
    print("=" * 80)

