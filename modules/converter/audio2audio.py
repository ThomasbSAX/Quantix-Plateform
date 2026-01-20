"""
Module de conversion audio robuste.

Expose `convert(input_audio, output_audio, ...)` avec :
- backend pydub (si disponible) pour une API Python simple
- fallback ffmpeg via subprocess
- diagnostics précis (backend, codec, ffmpeg manquant, etc.)
"""

from __future__ import annotations

import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional, Union, List, Literal


Logger = logging.getLogger(__name__)

Backend = Literal["auto", "pydub", "ffmpeg"]

# Formats audio supportés (pydub + ffmpeg)
SUPPORTED_INPUT_FORMATS = {
    # Formats non compressés
    "wav", "wave", "aiff", "aif", "aifc",
    # Formats compressés avec perte
    "mp3", "mp4", "m4a", "aac", "ac3",
    "ogg", "oga", "opus", "spx",
    "wma", "wmv",
    # Formats compressés sans perte
    "flac", "alac", "ape", "wv", "tta",
    # Formats raw et autres
    "raw", "pcm", "au", "snd",
    "amr", "3gp", "3gpp",
    "mka", "webm",
    # Formats vidéo (extraction audio)
    "mp4", "avi", "mkv", "mov", "flv", "webm", "mpg", "mpeg",
}

SUPPORTED_OUTPUT_FORMATS = {
    "wav", "mp3", "flac", "ogg", "opus", "m4a", "aac", 
    "wma", "aiff", "au", "ac3", "webm", "amr"
}


class AudioConversionError(RuntimeError):
    pass


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioConversionError(
            "ffmpeg introuvable. Installez ffmpeg (brew/apt) ou utilisez pydub avec ffmpeg disponible."
        )
    return ffmpeg


def _run_ffmpeg(
    *,
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    sample_rate: Optional[int],
    bitrate: Optional[str],
    channels: Optional[int],
    codec: Optional[str],
    overwrite: bool,
) -> None:
    cmd: List[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(input_path)]

    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    if bitrate is not None:
        cmd += ["-b:a", bitrate]
    if codec is not None:
        cmd += ["-c:a", codec]
    if overwrite:
        cmd.append("-y")

    cmd.append(str(output_path))

    Logger.debug("ffmpeg command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        raise AudioConversionError(f"ffmpeg a échoué:\n{err}") from exc


def _run_pydub(
    *,
    input_path: Path,
    output_path: Path,
    sample_rate: Optional[int],
    bitrate: Optional[str],
    channels: Optional[int],
    codec: Optional[str],
    overwrite: bool,
) -> None:
    try:
        from pydub import AudioSegment  # type: ignore
    except ModuleNotFoundError as exc:
        raise AudioConversionError(
            "pydub non installé. `pip install pydub` (ffmpeg requis)."
        ) from exc

    audio = AudioSegment.from_file(str(input_path))

    if sample_rate is not None:
        audio = audio.set_frame_rate(sample_rate)
    if channels is not None:
        audio = audio.set_channels(channels)

    if overwrite and output_path.exists():
        output_path.unlink(missing_ok=True)

    fmt = output_path.suffix.lstrip(".").lower() or None

    parameters: List[str] = []
    if codec is not None:
        parameters += ["-c:a", codec]

    export_kwargs = {}
    if bitrate is not None:
        export_kwargs["bitrate"] = bitrate

    try:
        audio.export(
            str(output_path),
            format=fmt,
            parameters=parameters or None,
            **export_kwargs,
        )
    except Exception as exc:
        raise AudioConversionError(f"pydub a échoué: {exc}") from exc


def convert(
    input_audio: Union[str, Path],
    output_audio: Union[str, Path],
    *,
    sample_rate: Optional[int] = None,
    bitrate: Optional[str] = None,
    channels: Optional[int] = None,
    codec: Optional[str] = None,
    overwrite: bool = True,
    backend: Backend = "auto",
) -> None:
    """
    Convertit un fichier audio.

    Formats d'entrée supportés:
    - Non compressés: WAV, AIFF, AU
    - Compressés avec perte: MP3, AAC, M4A, OGG, OPUS, WMA
    - Compressés sans perte: FLAC, ALAC, APE, WV, TTA
    - Autres: AMR, 3GP, WebM, MKA
    - Extraction audio depuis vidéo: MP4, AVI, MKV, MOV, FLV, etc.

    Formats de sortie supportés:
    WAV, MP3, FLAC, OGG, OPUS, M4A, AAC, WMA, AIFF, AU, AC3, WebM, AMR

    backend:
    - 'auto'   : pydub si dispo, sinon ffmpeg
    - 'pydub'  : force pydub (erreur si indisponible)
    - 'ffmpeg' : force ffmpeg
    """

    input_path = Path(input_audio)
    output_path = Path(output_audio)

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier source introuvable: {input_path}")

    # Validation des formats
    input_ext = input_path.suffix.lstrip(".").lower()
    output_ext = output_path.suffix.lstrip(".").lower()
    
    if input_ext and input_ext not in SUPPORTED_INPUT_FORMATS:
        Logger.warning(
            f"Format d'entrée '{input_ext}' non listé dans les formats supportés. "
            f"Tentative de conversion quand même..."
        )
    
    if output_ext and output_ext not in SUPPORTED_OUTPUT_FORMATS:
        Logger.warning(
            f"Format de sortie '{output_ext}' non listé dans les formats supportés. "
            f"Tentative de conversion quand même..."
        )

    if backend not in {"auto", "pydub", "ffmpeg"}:
        raise ValueError(f"backend invalide: {backend}")

    last_error: Optional[Exception] = None

    if backend in {"auto", "pydub"}:
        try:
            _run_pydub(
                input_path=input_path,
                output_path=output_path,
                sample_rate=sample_rate,
                bitrate=bitrate,
                channels=channels,
                codec=codec,
                overwrite=overwrite,
            )
            return
        except Exception as exc:
            last_error = exc
            if backend == "pydub":
                raise
            Logger.debug("pydub indisponible ou en échec, fallback ffmpeg: %s", exc)

    ffmpeg = _require_ffmpeg()
    try:
        _run_ffmpeg(
            ffmpeg=ffmpeg,
            input_path=input_path,
            output_path=output_path,
            sample_rate=sample_rate,
            bitrate=bitrate,
            channels=channels,
            codec=codec,
            overwrite=overwrite,
        )
    except Exception as exc:
        if last_error:
            raise AudioConversionError(
                f"Tous les backends ont échoué.\n"
                f"- pydub: {last_error}\n"
                f"- ffmpeg: {exc}"
            ) from exc
        raise


convert_audio = convert

__all__ = [
    "convert", 
    "convert_audio", 
    "AudioConversionError",
    "SUPPORTED_INPUT_FORMATS",
    "SUPPORTED_OUTPUT_FORMATS",
]
