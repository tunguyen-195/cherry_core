from abc import ABC, abstractmethod
from .asr_port import ITranscriber
from .llm_port import ILLMEngine
from .correction_port import ITextCorrector

class ISystemFactory(ABC):
    @abstractmethod
    def create_transcriber(self, model_name: str | None = None) -> ITranscriber:
        pass
        
    @abstractmethod
    def create_llm_engine(self) -> ILLMEngine:
        pass

    @abstractmethod
    def create_corrector(self) -> 'ITextCorrector':
        pass
