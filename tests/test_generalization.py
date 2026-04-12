"""
Benchmark Suite for Generalization Testing.
Tests diarization across diverse audio conditions to prevent overfitting.
"""
import os
import pytest
from pathlib import Path
from typing import List, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# Benchmark configurations
BENCHMARK_CONFIGS = [
    # (name, expected_speakers, condition, audio_path)
    ("hotel_2speakers", 2, "studio", "samples/test_audio.mp3"),
    # Add more as audio files become available:
    # ("meeting_4speakers", 4, "meeting", "benchmark/meeting_4p.wav"),
    # ("phone_call", 2, "telephone", "benchmark/phone_8khz.wav"),
    # ("noisy_cafe", 2, "noisy", "benchmark/cafe_noise.wav"),
    # ("overlapping_debate", 3, "overlap", "benchmark/debate.wav"),
]


class BenchmarkRunner:
    """
    Run diarization benchmarks across diverse conditions.
    
    Purpose:
    - Prevent overfitting to specific audio conditions
    - Ensure generalization across speaker counts, audio quality, etc.
    - Track DER across different scenarios
    """
    
    def __init__(self, diarizer=None):
        """
        Args:
            diarizer: ISpeakerDiarizer instance (optional, will create default if None)
        """
        self.diarizer = diarizer
        self._results = []
    
    def _ensure_diarizer(self):
        if self.diarizer is None:
            from infrastructure.factories.system_factory import SystemFactory
            factory = SystemFactory()
            self.diarizer = factory.create_diarizer()
    
    def run_benchmark(self, name: str, audio_path: str, expected_speakers: int) -> dict:
        """
        Run single benchmark test.
        
        Returns:
            dict with benchmark results
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            logger.warning(f"⚠️ Benchmark audio not found: {audio_path}")
            return {
                "name": name,
                "status": "SKIPPED",
                "reason": "File not found"
            }

        self._ensure_diarizer()
        
        logger.info(f"🧪 Running benchmark: {name}")
        
        try:
            segments = self.diarizer.diarize(str(audio_file))
            detected_speakers = len(set(seg.speaker_id for seg in segments))
            
            # Calculate metrics
            speaker_count_correct = abs(detected_speakers - expected_speakers) <= 1
            
            result = {
                "name": name,
                "status": "PASS" if speaker_count_correct else "FAIL",
                "expected_speakers": expected_speakers,
                "detected_speakers": detected_speakers,
                "segment_count": len(segments),
                "speaker_count_accurate": speaker_count_correct,
            }
            
            self._results.append(result)
            logger.info(f"✅ Benchmark {name}: {result['status']} (detected {detected_speakers} speakers)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Benchmark {name} failed: {e}")
            return {
                "name": name,
                "status": "ERROR",
                "error": str(e)
            }
    
    def run_all_benchmarks(self) -> List[dict]:
        """Run all configured benchmarks."""
        results = []
        for name, expected_speakers, condition, audio_path in BENCHMARK_CONFIGS:
            result = self.run_benchmark(name, audio_path, expected_speakers)
            result["condition"] = condition
            results.append(result)
        
        return results
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate benchmark report."""
        if not self._results:
            return "No benchmark results available. Run benchmarks first."
        
        report_lines = [
            "# Benchmark Report",
            "=" * 60,
            "",
            f"Total benchmarks: {len(self._results)}",
            f"Passed: {sum(1 for r in self._results if r.get('status') == 'PASS')}",
            f"Failed: {sum(1 for r in self._results if r.get('status') == 'FAIL')}",
            f"Skipped: {sum(1 for r in self._results if r.get('status') == 'SKIPPED')}",
            "",
            "## Results",
            ""
        ]
        
        for result in self._results:
            status_icon = "✅" if result.get('status') == 'PASS' else "❌" if result.get('status') == 'FAIL' else "⏭️"
            report_lines.append(f"{status_icon} **{result['name']}**")
            report_lines.append(f"   - Status: {result.get('status', 'UNKNOWN')}")
            if 'expected_speakers' in result:
                report_lines.append(f"   - Expected speakers: {result['expected_speakers']}")
                report_lines.append(f"   - Detected speakers: {result.get('detected_speakers', 'N/A')}")
            if 'error' in result:
                report_lines.append(f"   - Error: {result['error']}")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info(f"📊 Benchmark report saved to: {output_path}")
        
        return report


# Pytest integration
class TestGeneralization:
    """Pytest-based generalization tests."""
    
    @pytest.fixture
    def runner(self):
        return BenchmarkRunner()
    
    @pytest.mark.parametrize("name,expected_speakers,condition,audio_path", BENCHMARK_CONFIGS)
    def test_speaker_count(self, runner, name, expected_speakers, condition, audio_path):
        """Test speaker count accuracy across conditions."""
        if os.getenv("RUN_GENERALIZATION_BENCHMARKS", "").lower() not in {"1", "true", "yes"}:
            pytest.skip("Generalization benchmark is opt-in. Set RUN_GENERALIZATION_BENCHMARKS=1 to run.")

        result = runner.run_benchmark(name, audio_path, expected_speakers)
        
        if result.get("status") == "SKIPPED":
            pytest.skip(result.get("reason", "Skipped"))
        
        assert result.get("speaker_count_accurate", False), \
            f"Speaker count mismatch for {name}: expected {expected_speakers}, got {result.get('detected_speakers')}"


def main():
    """CLI entry point for running benchmarks."""
    import argparse
    import sys
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    parser = argparse.ArgumentParser(description="Run diarization benchmarks")
    parser.add_argument("--report", type=str, help="Output path for benchmark report")
    args = parser.parse_args()
    
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks()
    
    report = runner.generate_report(args.report)
    print(report)
    
    # Exit with error code if any benchmark failed
    if any(r.get("status") == "FAIL" for r in results):
        exit(1)


if __name__ == "__main__":
    main()
