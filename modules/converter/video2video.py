from __future__ import annotations
from pathlib import Path
import subprocess
from typing import Optional


def video_to_video(
    input_video: str | Path,
    output_video: str | Path,
    *,
    video_codec: Optional[str] = None,
    audio_codec: Optional[str] = None,
    crf: Optional[int] = None,
    preset: str = "slow",
    fps: Optional[int] = None,
    resolution: Optional[str] = None,  # "1920x1080"
    lossless: bool = False,
    overwrite: bool = True,
) -> Path:
    """
    Conversion vidéo → vidéo avec qualité maximale.

    Règles :
    - stream copy par défaut (aucune perte)
    - re-encodage uniquement si explicitement demandé
    - paramètres FFmpeg propres et non destructifs
    """

    input_video = Path(input_video)
    output_video = Path(output_video)

    if not input_video.exists():
        raise FileNotFoundError(input_video)

    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_video),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-map_metadata", "0",
    ]

    # ---------- Vidéo ----------
    if lossless:
        cmd += ["-c:v", "libx264rgb", "-crf", "0"]
    elif video_codec:
        cmd += ["-c:v", video_codec]
        if crf is not None:
            cmd += ["-crf", str(crf), "-preset", preset]
    else:
        cmd += ["-c:v", "copy"]

    # ---------- Audio ----------
    if audio_codec:
        cmd += ["-c:a", audio_codec]
    else:
        cmd += ["-c:a", "copy"]

    # ---------- Filtres ----------
    vf = []

    if resolution:
        vf.append(f"scale={resolution}")

    if fps:
        vf.append(f"fps={fps}")

    if vf:
        cmd += ["-vf", ",".join(vf)]

    # ---------- Overwrite ----------
    if overwrite:
        cmd.append("-y")

    cmd.append(str(output_video))

    subprocess.run(cmd, check=True)
    return output_video
