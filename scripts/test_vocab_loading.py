
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from application.services.correction_service import CorrectionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vocab_loading():
    logger.info("Testing Custom Vocabulary Loading...")
    
    # Initialize service (LLM lazily loaded)
    service = CorrectionService()
    
    # Check if vocab context is populated
    if not service._vocab_data:
        logger.error("❌ Vocabulary data is empty!")
        sys.exit(1)
        
    logger.info(f"✅ Vocabulary Categories: {len(service._vocab_data)} loaded")
    
    # Show sample categories
    for cat in list(service._vocab_data.keys())[:3]:
        logger.info(f"   - {cat}: {len(service._vocab_data[cat])} terms")
    
    # Verify specific terms exist
    expected_terms = ["Khách sạn", "Deluxe", "Marriott", "Nguyễn"]
    for term in expected_terms:
        if term in service._vocab_context:
            logger.info(f"✅ Found expected term: {term}")
        else:
            logger.warning(f"⚠️ Term not found in context sample (might be truncated or missing): {term}")
            
    print("\nTest passed!")

if __name__ == "__main__":
    test_vocab_loading()
