"""Speech-to-text.

Deux usages:
- `speech2txt()` : enregistrement micro en local (CLI) puis transcription.
- `AudioTranscriber` : utilisé par Flask pour transcrire un fichier audio/vidéo uploadé.

Important: les dépendances micro (sounddevice/scipy) sont importées *lazy* afin que
le serveur Flask puisse démarrer même si elles ne sont pas installées.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

try:
    from pydub import AudioSegment
except Exception as e:  # pragma: no cover
    AudioSegment = None
    _PYDUB_IMPORT_ERROR = e

def speech2txt():
    """Enregistre la voix et retourne la transcription"""

    # Imports lazy: uniquement nécessaires pour l'enregistrement micro local
    import numpy as np
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError("sounddevice est requis pour speech2txt() en local") from e
    try:
        from scipy.io.wavfile import write
    except Exception as e:
        raise RuntimeError("scipy est requis pour speech2txt() en local") from e
    
    # Configuration
    SAMPLE_RATE = 44100
    
    print("Enregistrement en cours...")
    print("Appuyez sur ENTRÉE pour arrêter l'enregistrement")
    
    # Variable pour contrôler l'enregistrement
    recording = []
    stop_recording = threading.Event()
    
    def record_audio():
        """Enregistre l'audio jusqu'à ce que stop_recording soit activé"""
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            while not stop_recording.is_set():
                data, _ = stream.read(SAMPLE_RATE // 10)
                recording.append(data)
    
    # Démarre l'enregistrement dans un thread
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()
    
    # Attend que l'utilisateur appuie sur Entrée
    input()
    
    # Arrête l'enregistrement
    stop_recording.set()
    record_thread.join()
    
    print("Enregistrement terminé !")
    
    # Combine tous les blocs audio
    audio_data = np.concatenate(recording, axis=0)
    
    # Sauvegarde temporaire
    temp_file = "temp_recording.wav"
    write(temp_file, SAMPLE_RATE, audio_data)
    
    # Transcription avec Whisper
    print("Transcription en cours...")
    from faster_whisper import WhisperModel
    model = WhisperModel("small")
    segments, info = model.transcribe(temp_file)
    
    text = " ".join([seg.text for seg in segments])
    
    # Nettoyage
    os.remove(temp_file)
    
    print("\nTranscription :")
    print(text)
    with open("transcription.txt", "w") as f:
        f.write(text)
    
    print("Transcription sauvegardée dans transcription.txt")
    
    return text


class AudioTranscriber:
    def __init__(self, model_size="small"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size)

    def _convert_to_wav(self, audio_path: str) -> str:
        if AudioSegment is None:
            raise RuntimeError(
                "pydub (et ffmpeg) est requis pour convertir l'audio sur ce serveur"
            ) from _PYDUB_IMPORT_ERROR

        audio = AudioSegment.from_file(audio_path)
        wav_path = str(Path(audio_path).with_suffix(".wav"))
        audio.export(wav_path, format="wav")
        return wav_path

    def transcribe(self, audio_path: str, write_txt=True, write_docx=False) -> str:
        wav = self._convert_to_wav(audio_path)
        segments, _ = self.model.transcribe(wav)
        text = "".join([s.text for s in segments])

        out_txt = None
        out_docx = None
        if write_txt:
            out_txt = str(Path(audio_path).with_suffix(".txt"))
            with open(out_txt, "w") as f:
                f.write(text)
        if write_docx:
            from docx import Document
            out_docx = str(Path(audio_path).with_suffix(".docx"))
            doc = Document()
            doc.add_paragraph(text)
            doc.save(out_docx)
        # Retourne le chemin du fichier généré (priorité docx)
        if write_docx:
            return out_docx
        elif write_txt:
            return out_txt
        return text
