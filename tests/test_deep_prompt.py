import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from application.services.prompt_manager import PromptManager

def test_deep_prompt_render():
    print("🧪 Testing Deep Prompt Architecture...")
    pm = PromptManager()
    
    transcript = "Alo, kèo này x2 tài khoản. Chúng tôi uy tín lắm."
    
    # Render Deep Template
    prompt = pm.render_prompt(
        "deep_investigation.j2", 
        transcript, 
        scenario="high_tech_fraud"
    )
    
    print("\n✅ Rendered Master Prompt:")
    print("-" * 40)
    print(prompt[:500] + "...") # Print start
    print("..." + prompt[-500:]) # Print end
    print("-" * 40)
    
    # Verify Modules are included
    assert "MODULE: SVA/CBCA" in prompt, "Missing SVA Module"
    assert "MODULE: SCAN" in prompt, "Missing SCAN Module"
    assert "MODULE: CRIMINAL_PSYCHOLOGY_VN" in prompt, "Missing Psychology Module"
    assert "MODULE: INTELLIGENCE_5W1H" in prompt, "Missing 5W1H Module"
    assert "MODULE: EMOTIONAL_SPECTRUM" in prompt, "Missing Emotions Module"
    assert "MODULE: QUANTITATIVE_DATA" in prompt, "Missing Quantitative Module"
    assert "MODULE: SENSITIVE_INTEL" in prompt, "Missing Sensitive Module"
    assert "lùa gà" in prompt, "Missing Slang Dictionary from Scenario"
    
    print("\n✨ Modular Architecture Verified Successfully!")

if __name__ == "__main__":
    test_deep_prompt_render()
