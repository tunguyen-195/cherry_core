import sys
try:
    from whisperx import DiarizationPipeline
    print("SUCCESS: from whisperx import DiarizationPipeline")
except ImportError:
    print("FAIL: from whisperx import DiarizationPipeline")
    
    try:
        from whisperx.diarize import DiarizationPipeline
        print("SUCCESS: from whisperx.diarize import DiarizationPipeline")
    except ImportError:
        print("FAIL: from whisperx.diarize import DiarizationPipeline")
        
    try:
        import whisperx.diarize
        print("whisperx.diarize attributes:", dir(whisperx.diarize))
    except ImportError:
        print("FAIL: import whisperx.diarize")
