
import sys
import os
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from application.services.correction_service import CorrectionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_retrieval_logic():
    logger.info("Testing Smart Vocabulary Retrieval...")
    
    service = CorrectionService()
    
    # Check if vocab loaded
    if not service._vocab_data:
        logger.error("❌ Vocabulary NOT loaded!")
        return

    test_cases = [
        {
            "transcript": "tôi muốn đặt phòng khách sạn tại đà nẵng",
            "expected_category": "hospitality",
            "expected_term": "Deluxe" # 'Deluxe' is in hospitality list
        },
        {
            "transcript": "công an bắt giữ đối tượng buôn bán ma túy đá",
            "expected_category": "drugs",
            "expected_term": "Methamphetamine" # from drugs list
        },
        {
            "transcript": "u là trời hôm nay xui quá đi",
            "expected_category": "slang_genz",
            "expected_term": "Ét o ét" # from slang list
        },
        {
            "transcript": "đồng bào dân tộc mông ăn tết độc lập",
            "expected_category": "ethnic_religion",
            "expected_term": "Nhà rông" # from ethnic list
        }
    ]
    
    for case in test_cases:
        transcript = case["transcript"]
        logger.info(f"\nScanning transcript: '{transcript}'")
        
        context = service._get_relevant_vocab(transcript)
        
        logger.info(f"Generated Context: {context[:100]}...")
        
        if case["expected_term"] in context:
            logger.info(f"✅ Success! Found expected term '{case['expected_term']}' in context.")
        else:
            logger.warning(f"❌ Failed! Did NOT find '{case['expected_term']}' in context.")
            
    print("\nRetrieval Logic Test Complete.")

if __name__ == "__main__":
    test_retrieval_logic()
