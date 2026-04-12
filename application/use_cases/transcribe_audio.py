from core.ports.asr_port import ITranscriber
from core.ports.correction_port import ITextCorrector
from core.domain.entities import Transcript
from typing import Optional

class TranscribeAudioUseCase:
    def __init__(self, transcriber: ITranscriber, corrector: Optional[ITextCorrector] = None):
        self.transcriber = transcriber
        self.corrector = corrector

    def execute(self, audio_path: str) -> Transcript:
        # 1. Transcribe
        transcript = self.transcriber.transcribe(audio_path)
        
        # 2. Correct (Optional)
        if self.corrector:
            original_text = transcript.text
            corrected_text = self.corrector.correct(original_text)
            transcript.text = corrected_text
            transcript.metadata['original_text'] = original_text
            transcript.metadata['corrected'] = True
            
        return transcript
