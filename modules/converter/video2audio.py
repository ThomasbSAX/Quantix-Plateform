from __future__ import annotations
from pathlib import Path
import subprocess
from typing import Optional


def video_to_audio(
    input_video: str | Path,
    output_audio: str | Path,
    *,
    codec: Optional[str] = None,
    bitrate: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    lossless: bool = False,
    overwrite: bool = True,
) -> Path:
    """
    Extrait l'audio d'une vidéo avec qualité maximale.

    Stratégie :
    - copie du flux audio si possible
    - sinon re-encodage propre (lossless ou high quality)
    """

    input_video = Path(input_video)
    output_audio = Path(output_audio)

    if not input_video.exists():
        raise FileNotFoundError(input_video)

    output_audio.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(input_video), "-vn"]

    if codec:
        cmd += ["-c:a", codec]
    elif lossless:
        # choix lossless par défaut
        cmd += ["-c:a", "flac"]
    else:
        # copie directe si possible
        cmd += ["-c:a", "copy"]

    if bitrate and not lossless:
        cmd += ["-b:a", bitrate]

    if sample_rate:
        cmd += ["-ar", str(sample_rate)]

    if channels:
        cmd += ["-ac", str(channels)]

    if overwrite:
        cmd.append("-y")

    cmd.append(str(output_audio))

    subprocess.run(cmd, check=True)

    return output_audio
