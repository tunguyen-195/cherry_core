# Diarization Module
# Modular speaker diarization for Cherry Core V2

__all__: list[str] = []

try:
    from infrastructure.adapters.diarization.resemblyzer_adapter import ResemblyzerAdapter
except Exception:
    ResemblyzerAdapter = None  # type: ignore[assignment]
else:
    __all__.append("ResemblyzerAdapter")

# Future imports as they become available:
# from infrastructure.adapters.diarization.enhanced_adapter import EnhancedDiarizer
# from infrastructure.adapters.diarization.nemo_adapter import NeMoDiarizer
