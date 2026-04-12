import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from application.services.prompt_manager import PromptManager

def test_prompt_rendering():
    print("🧪 Testing Prompt Manager...")
    pm = PromptManager()
    
    transcript = "Alo, tối nay đi bay không? Nhớ mang kẹo nhé. Anh cần 5 lít cơm."
    
    # Render with scenario
    prompt = pm.render_prompt("investigation.j2", transcript, scenario="drug_trafficking")
    
    print("\n✅ Rendered Prompt Preview:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # Assertions
    assert "TỪ ĐIỂN MẬT NGỮ" in prompt
    assert "kẹo" in prompt
    assert "thuốc lắc" in prompt
    assert "cơm" in prompt
    assert "hàng trắng" in prompt
    assert transcript in prompt
    
    print("\n✨ Prompt Logic Verified Successfully!")

if __name__ == "__main__":
    test_prompt_rendering()
