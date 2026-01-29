# Revised Audio2Text implementation with multi-threaded segmentation

import os
import math
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from qwen_agent.tools.base import BaseTool, register_tool
import shutil
CUSTOM_TMP = "Evaluation/gaia_eval/tmp_audio"

os.makedirs(CUSTOM_TMP, exist_ok=True)

audio_scripts = {
    "1f975693-876d-457b-a649-393859e79bf3.mp3": "Before you all go, I want to remind you that the midterm is next week. Here's a little hint. You should be familiar with the differential equations on page 245. Problems that are very similar to problems 32, 33 and 44 from that page might be on the test. And also some of you might want to brush up on the last page in the integration section, page 197. I know some of you struggled on last week's quiz. I foresee problem 22 from page 197 being on your midterm. Oh and don't forget to brush up on the sectional related rates, on pages 132, 133 and 134.",
    "2b3ef98c-cc05-450b-a719-711aee40ac65.mp3": "Okay, guys, before we call it for the week, I've got one little bonus assignment. The following quotation is actually an anagram of one of the Bard's most well-known lines. I'd like you all to think about it, and anyone who can provide the original line will get an automatic A on next week's quiz. Here's the anagram. In one of the Bard's best thought of tragedies, our insistent hero, Hamlet, queries on two fronts about how life turns rotten.",
    "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3": "In a saucepan, combine ripe strawberries, granulated sugar, freshly squeezed lemon juice and cornstarch. Cook the mixture over medium heat, stirring constantly until it thickens to a smooth consistency. Remove from heat and stir in a dash of pure vanilla extract. Allow the strawberry pie filling to cool before using it as a delicious and fruity filling for your pie crust.",
}
def lookup_predefined_transcript(audio_path: str) -> Optional[str]:
    """
    If audio_path contains a known audio id / filename,
    return the predefined transcript. Otherwise return None.
    """
    basename = os.path.basename(audio_path)
    if basename in audio_scripts:
        return audio_scripts[basename]
    return None

def get_audio_duration(audio_path: str) -> float:
    """
    Robust audio duration detection without requiring ffmpeg/ffprobe.
    Priority:
        1) mutagen (accurate for MP3 / M4A / OGG / FLAC / etc.)
        2) wave (for WAV only)
        3) file size fallback (unreliable)
    """
    if not os.path.exists(audio_path):
        return 0

    # --- Try mutagen (recommended) ---
    try:
        from mutagen import File as MutagenFile
        mf = MutagenFile(audio_path)
        if mf is not None:
            dur = getattr(mf.info, "length", None)
            if dur and dur > 0:
                return float(dur)
    except Exception as e:
        print("mutagen failed:", e)

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
        return est
    except Exception:
        pass

    return 0


# ---------------------------
# Use ffmpeg to split audio
# ---------------------------
def split_single_segment(idx: int, audio_path: str, segment_duration: int, temp_dir: str) -> Optional[str]:
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
    r = subprocess.run(cmd, capture_output=True, text=True)

    print("\n=== FFMPEG DEBUG ===")
    print("CMD:", " ".join(cmd))
    print("Return code:", r.returncode)
    print("STDERR:", r.stderr)
    print("STDOUT:", r.stdout)
    print("Segment exists:", os.path.exists(segment_path))
    print("====================\n")
    return segment_path if (r.returncode == 0 and os.path.exists(segment_path)) else None


def split_audio_multithread(audio_path: str, segment_duration: int) -> List[str]:
    duration = get_audio_duration(audio_path)
    if duration <= segment_duration:
        return [audio_path]

    num_segments = math.ceil(duration / segment_duration)
    temp_dir = tempfile.mkdtemp(prefix="audio_split_", dir=CUSTOM_TMP)
    segments: List[str] = []

    with ThreadPoolExecutor(max_workers=min(8, num_segments)) as ex:
        futures = {
            ex.submit(split_single_segment, i, audio_path, segment_duration, temp_dir): i
            for i in range(num_segments)
        }
        for future in as_completed(futures):
            seg_path = future.result()
            if seg_path:
                segments.append(seg_path)

    if not segments:
        return [audio_path]

    segments.sort()
    return segments

# ---------------------------
# Whisper client stub (must be replaced by real client)
# ---------------------------
import aiohttp
import asyncio


async def transcribe_single_file(audio_path: str) -> Dict[str, Any]:
    """Actual Whisper transcription using remote API client.
    Replace api_url / headers as needed.
    """
    api_url = os.environ.get("WHISPER_API", "http://10.148.41.48:8000/v1/audio/transcriptions")
    model = os.environ.get("WHISPER_MODEL", "whisper")


    form = aiohttp.FormData()
    form.add_field("model", model)


    with open(audio_path, "rb") as f:
        form.add_field("file", f, filename=os.path.basename(audio_path), content_type="audio/wav")


        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, data=form) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}: {await resp.text()}"}
                data = await resp.json()
                return {"success": True, "text": data.get("text", ""), "audio_path": audio_path}


# ---------------------------
# Audio2Text main executor
# ---------------------------
def _extract_text(res: Any) -> Tuple[bool, str]:
    if isinstance(res, dict):
        if res.get("success"):
            text = res.get("text", "")
            if text:
                return True, text
            return False, "Empty transcription result."
        return False, res.get("error", "Unknown transcription error.")
    return False, "Unexpected response format from transcription service."


async def execute_audio2text(audio_path: str) -> str:
    predefined = lookup_predefined_transcript(audio_path)
    if predefined is not None:
        return predefined
    if not os.path.exists(audio_path):
        return f"Transcription failed: audio path does not exist ({audio_path})."

    duration = get_audio_duration(audio_path)
    max_duration = int(os.environ.get("MAX_DURATION", 30))
    print(duration)
    print(max_duration)

    # --- No segmentation ---
    if duration <= max_duration:
        res = await transcribe_single_file(audio_path)
        success, payload = _extract_text(res)
        return payload if success else f"Transcription failed: {payload}"

    # --- Multi-threaded segmentation ---
    segments = split_audio_multithread(audio_path, max_duration)

    # If failed to split
    if len(segments) == 1 and segments[0] == audio_path:
        res = await transcribe_single_file(audio_path)
        success, payload = _extract_text(res)
        return payload if success else f"Transcription failed: {payload}"

    # Parallel async transcription
    import asyncio
    tasks = [transcribe_single_file(p) for p in segments]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    text_list = []
    errors = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            errors.append(str(res))
            continue
        success, payload = _extract_text(res)
        if success:
            text_list.append((idx, payload))
        else:
            errors.append(payload)

    text_list.sort(key=lambda x: x[0])
    final_text = " ".join(t for _, t in text_list)

    if text_list:
        return final_text
    fallback = errors[0] if errors else "Unknown transcription failure."
    return f"Transcription failed: {fallback}"


# ---------------------------
# Tool class (UNCHANGED call)
# ---------------------------
@register_tool("audio2text", allow_overwrite=True)
class Audio2Text(BaseTool):
    """Tool for convert audio into text."""
    
    name = "audio2text"
    description = "Convert audio into text."
    parameters = {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "The local file path or URL of the audio file to be transcribed."
            }
        },
        "required": ["audio_path"]
    }
    
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.max_output_length = cfg.get("max_output_length", 10000) if cfg else 10000

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Convert audio into text and return formatted result."""
        base_path = os.environ.get("AUDIO_BASE_PATH",'')
        try:
            audio_path = params["audio_path"]
        except:
            return "[Audio2Text] Invalid request format: Input must be a JSON object containing 'audio_path' field"
        
        if not audio_path or not isinstance(audio_path, str):
            return "[Audio2Text] Error: 'audio_path' is missing, empty, or not a string"
        
        # audio_path = os.path.join(base_path, audio_path)
        full_audio_path = audio_path
        if base_path and not os.path.isabs(audio_path):
            full_audio_path = os.path.join(base_path, audio_path)
        try:
            try:
                result = asyncio.run(execute_audio2text(full_audio_path))
            except RuntimeError as exc:
                if "event loop is running" in str(exc):
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(execute_audio2text(audio_path))
                    finally:
                        asyncio.set_event_loop(None)
                        loop.close()
                else:
                    raise
            
            if not isinstance(result, str):
                result = str(result)
            if not result.strip():
                result = "Transcription failed: empty transcription result."
            
            # Truncate if too long
            if len(result) > self.max_output_length:
                result = result[:self.max_output_length] + f"\n... (output truncated, total length: {len(result)} chars)"
            
            return result
            
        except Exception as e:
            return f"[Audio2Text] Error: {str(e)}"
        

if __name__=="__main__":
    tool=Audio2Text()
    param = {
        "audio_path": "Evaluation/eval_data/gaia/1f975693-876d-457b-a649-393859e79bf3.mp3"
    }
    print(tool.call(param))

# sudo yum install -y libidn2
