"""
Cherry Core V2 - Diarization Test Script
Tests speaker diarization on user audio file.
"""
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.factories.system_factory import SystemFactory

# Test Audio
AUDIO_FILE = Path(r"E:\research\Cherry2\Tiếp nhận yêu cầu đặt phòng của khách lẻ qua điện thoại.mp3")

def main():
    print("=" * 60)
    print("🎤 CHERRY CORE V2 - SPEAKER DIARIZATION TEST")
    print("=" * 60)
    print(f"📁 Audio: {AUDIO_FILE.name}")
    print("=" * 60)
    
    if not AUDIO_FILE.exists():
        print(f"❌ File not found: {AUDIO_FILE}")
        return
    
    factory = SystemFactory()
    
    # Create diarizer (2 speakers: receptionist + customer)
    try:
        diarizer = factory.create_diarizer(n_speakers=2)
        print("✅ Diarizer initialized (Resemblyzer + Spectral Clustering)")
    except ImportError as e:
        print(f"❌ Diarizer not available: {e}")
        print("   Install: pip install resemblyzer scikit-learn")
        return
    
    # Run diarization
    print("\n🔊 Running diarization...")
    try:
        speaker_segments = diarizer.diarize(str(AUDIO_FILE))
        
        print(f"\n✅ Diarization complete: {len(speaker_segments)} segments detected")
        print("-" * 50)
        
        # Print first 10 segments
        for i, seg in enumerate(speaker_segments[:10]):
            print(f"  [{seg.start_time:6.1f}s - {seg.end_time:6.1f}s] {seg.speaker_id}")
        
        if len(speaker_segments) > 10:
            print(f"  ... and {len(speaker_segments) - 10} more segments")
        
        # Summary
        speakers = set(seg.speaker_id for seg in speaker_segments)
        print(f"\n📊 Summary: {len(speakers)} unique speakers detected")
        for speaker in sorted(speakers):
            count = sum(1 for seg in speaker_segments if seg.speaker_id == speaker)
            print(f"   {speaker}: {count} segments")
            
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
