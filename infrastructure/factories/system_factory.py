from core.ports.system_factory import ISystemFactory
from core.ports.asr_port import ITranscriber
from core.ports.llm_port import ILLMEngine
from core.ports.correction_port import ITextCorrector
from core.ports.diarization_port import ISpeakerDiarizer
import core.config as config


class SystemFactory(ISystemFactory):
    def create_transcriber(self, model_name: str = None) -> ITranscriber:
        model_name = model_name or config.ASRConfig.ENGINE

        if model_name == "phowhisper":
            from infrastructure.adapters.asr.phowhisper_adapter import PhoWhisperAdapter

            return PhoWhisperAdapter()
        if model_name == "whisper-v3":
            from infrastructure.adapters.asr.whisperv3_adapter import WhisperV3Adapter

            return WhisperV3Adapter()

        from infrastructure.adapters.asr.whisperv2_adapter import WhisperV2Adapter

        return WhisperV2Adapter()

    def create_llm_engine(self) -> ILLMEngine:
        import logging

        logger = logging.getLogger(__name__)

        if config.USE_VLLM:
            try:
                from infrastructure.adapters.llm.vllm_adapter import VLLMAdapter

                logger.info("Attempting to load vLLM engine...")
                adapter = VLLMAdapter()
                if adapter.load():
                    logger.info("vLLM engine loaded successfully.")
                    return adapter
                logger.warning("vLLM load() returned False. Falling back to Llama.cpp.")
            except Exception as e:
                logger.warning(f"vLLM initialization failed ({e}). Falling back to Llama.cpp.")

        from infrastructure.adapters.llm.llamacpp_adapter import LlamaCppAdapter

        adapter = LlamaCppAdapter()
        if not adapter.load():
            raise RuntimeError("No LLM engine could be loaded.")
        return adapter

    def create_corrector(self) -> ITextCorrector:
        from infrastructure.adapters.correction.protonx_adapter import ProtonXAdapter

        return ProtonXAdapter()

    def create_diarizer(self, mode: str = None, n_speakers: int = None) -> ISpeakerDiarizer:
        import logging

        logger = logging.getLogger(__name__)
        engine = config.DiarizationConfig.ENGINE
        logger.info(f"Creating diarizer with engine: {engine}")

        if engine == "pyannote":
            if not config.DiarizationConfig.HF_TOKEN:
                logger.warning("HF_TOKEN not configured. Falling back to SpeechBrain diarization.")
                engine = "speechbrain"
            else:
                try:
                    from infrastructure.adapters.diarization.pyannote_adapter import PyannoteAdapter

                    return PyannoteAdapter(
                        hf_token=config.DiarizationConfig.HF_TOKEN,
                        num_speakers=n_speakers,
                    )
                except ImportError as e:
                    logger.warning(f"Pyannote not available ({e}). Falling back to SpeechBrain.")
                    engine = "speechbrain"

        if engine == "speechbrain":
            try:
                from infrastructure.adapters.diarization.speechbrain_adapter import SpeechBrainAdapter

                return SpeechBrainAdapter(n_speakers=n_speakers, use_vad=False)
            except ImportError as e:
                logger.warning(f"SpeechBrain not available ({e}). Falling back to legacy diarizers.")

        if mode == "enhanced":
            try:
                from infrastructure.adapters.diarization.enhanced_adapter import EnhancedDiarizer

                return EnhancedDiarizer(n_speakers=n_speakers)
            except ImportError as e:
                logger.warning(f"Enhanced diarizer not available ({e}). Falling back to Resemblyzer.")

        try:
            from infrastructure.adapters.diarization.resemblyzer_adapter import ResemblyzerAdapter

            return ResemblyzerAdapter(n_speakers=n_speakers or 2)
        except ImportError as e:
            raise RuntimeError(f"No diarization backend could be loaded: {e}") from e

    def create_speaker_refiner(self):
        from infrastructure.adapters.correction.contextual_refiner import ContextualSpeakerRefiner

        llm_engine = self.create_llm_engine()
        return ContextualSpeakerRefiner(llm_adapter=llm_engine)
