import logging
from pathlib import Path
from core.ports.asr_port import ITranscriber

logger = logging.getLogger(__name__)

class TranscriptionService:
    """
    Application Service for Transcription.
    Orchestrates the conversion of Audio to Text using Infrastructure Adapters.
    """
    def __init__(self, transcriber: ITranscriber):
        self.transcriber = transcriber
        
    def transcribe_file(self, file_path: str, use_vad: bool = True) -> str:
        """
        Transcribe audio file. 
        Note: VAD logic should ideally be inside the adapter or a separate VAD service.
        For now, we rely on the adapter's implementation or internal VAD.
        """
        logger.info(f"Requests transcription for: {file_path}")
        # The new adapters (WhisperV3, etc.) return a Transcript entity
        transcript_entity = self.transcriber.transcribe(file_path)
        return transcript_entity.text
