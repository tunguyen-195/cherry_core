import sys
import os

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infrastructure.factories.system_factory import SystemFactory

def main():
    print("🍒 Cherry Quick Correction Check")
    print("=" * 50)
    
    # Snippet from PhoWhisper Transcript
    raw_text = "bên em thì vẫn còn phòng đình lắc với giá là từ ba triệu đến ba triệu năm trăm nghìn và vào x kia tiếp với giá là bốn triệu năm trăm nghìn"
    
    print(f"🔴 ORIGINAL:\n{raw_text}\n")
    
    factory = SystemFactory()
    corrector = factory.create_corrector() # Loads ProtonX Legal
    
    corrected_text = corrector.correct(raw_text)
    
    print(f"🟢 CORRECTED:\n{corrected_text}")

if __name__ == "__main__":
    main()
