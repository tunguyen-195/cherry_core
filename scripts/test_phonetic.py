"""Quick test for phonetic corrector."""
import sys
sys.path.insert(0, ".")

from application.services.phonetic_corrector import get_phonetic_corrector

c = get_phonetic_corrector()
print(f"Loaded {c.get_correction_count()} patterns")

test = "đi lắc phòng vòng x kế tiếp G.W.Mirrors yêu đãi căn cướp"
result = c.correct(test)
print(f"Input:  {test}")
print(f"Output: {result}")
