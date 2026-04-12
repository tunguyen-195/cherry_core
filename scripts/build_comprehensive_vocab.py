
import os
import json
import logging
import urllib.request
from typing import Set, Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target file paths
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "vocab")
OUTPUT_FILE = os.path.join(ASSETS_DIR, "vn_general_vocabulary.json")

# Source URLs (Raw content)
SOURCES = {
    "Viet74K": "https://raw.githubusercontent.com/duyet/vietnamese-wordlist/master/Viet74K.txt",
    "VnStopwords": "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt",
    # Add more sources as discovered
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_text(url: str) -> List[str]:
    """Download text file from URL and return list of lines."""
    try:
        logger.info(f"Downloading from {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            logger.info(f"Downloaded {len(lines)} lines.")
            return lines
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return []

def build_vocabulary():
    ensure_dir(ASSETS_DIR)
    
    full_vocab: Set[str] = set()
    categories: Dict[str, List[str]] = {}

    # 1. Download and Process Viet74K
    viet74k = download_text(SOURCES["Viet74K"])
    if viet74k:
        full_vocab.update(viet74k)
        categories["general"] = viet74k
    
    # 2. Add existing custom vocabulary if exists
    custom_vocab_path = os.path.join(ASSETS_DIR, "vn_custom_vocabulary.json")
    if os.path.exists(custom_vocab_path):
        try:
            with open(custom_vocab_path, "r", encoding="utf-8") as f:
                custom_data = json.load(f)
                
            # Extract terms from custom structure
            if "hospitality" in custom_data:
                hospitality_terms =  custom_data["hospitality"].get("terms", []) + \
                                   custom_data["hospitality"].get("room_types", []) + \
                                   custom_data["hospitality"].get("hotel_names", [])
                full_vocab.update(hospitality_terms)
                categories["hospitality"] = hospitality_terms
                
            if "names" in custom_data:
                names = custom_data["names"].get("surnames", []) + \
                        custom_data["names"].get("middle_names", []) + \
                        custom_data["names"].get("given_names", [])
                full_vocab.update(names)
                categories["names"] = names
                
            logger.info("Merged existing custom vocabulary.")
            
        except Exception as e:
            logger.error(f"Failed to merge custom vocabulary: {e}")

    # 3. Add Manual Specialized Vocab (Security, Crime, Drugs, Ethnic/Religion)
    # Sourced from deep research (Laws, Criminology dictionaries)
    specialized_vocab = {
        "security_police": [
            "Công an", "Cảnh sát", "An ninh", "Trinh sát", "Hình sự", "Điều tra viên", 
            "Viện kiểm sát", "Tòa án", "Thẩm phán", "Luật sư", "Bị can", "Bị cáo", 
            "Tội phạm", "Khởi tố", "Bắt giữ", "Tạm giam", "Tạm giữ", "Truy nã", 
            "Đặc nhiệm", "Cảnh sát cơ động", "Phòng chống tội phạm", "An ninh mạng",
            "Phản động", "Khủng bố", "Bạo loạn", "Biểu tình", "Gây rối", "Trật tự công cộng",
            "Vũ khí", "Vật liệu nổ", "Công cụ hỗ trợ", "Trấn áp", "Mật phục", "Chuyên án"
        ],
        "drugs": [
            "Ma túy", "Thuốc phiện", "Cần sa", "Heroin", "Morphin", "Cocain", 
            "Ketamin", "Methamphetamine", "Ma túy đá", "Hồng phiến", "Thuốc lắc", 
            "Cỏ Mỹ", "Bóng cười", "Tiền chất", "Chất gây nghiện", "Chất hướng thần",
            "Cai nghiện", "Tàng trữ", "Vận chuyển", "Buôn bán", "Sử dụng trái phép",
            "Ngáo đá", "Sốc thuốc", "Đường dây", "Ổ nhóm", "Tụ điểm"
        ],
        "economic_crime": [
            "Tham ô", "Tham nhũng", "Hối lộ", "Rửa tiền", "Trốn thuế", "Buôn lậu",
            "Gian lận thương mại", "Hàng giả", "Đa cấp", "Lừa đảo", "Chiếm đoạt tài sản",
            "Vi phạm quy định", "Đấu thầu", "Ngân hàng", "Chứng khoán", "Tín dụng đen",
            "Cho vay nặng lãi", "Đòi nợ thuê", "Hợp đồng khống", "Khai khống"
        ],
        "ethnic_religion": [
            "Dân tộc", "Kinh", "Tày", "Thái", "Mường", "Khmer", "H'Mông", "Nùng", "Dao",
            "Gia Rai", "Ê Đê", "Ba Na", "Xơ Đăng", "Sán Chay", "Cơ Ho", "Chăm", "Sán Dìu",
            "Phật giáo", "Công giáo", "Tin lành", "Cao Đài", "Hòa Hảo", "Hồi giáo",
            "Tín ngưỡng", "Tôn giáo", "Thờ cúng", "Lễ hội", "Tuyên truyền", "Xúi giục",
            "Lợi dụng", "Chia rẽ", "Đoàn kết",
            "Giàng", "A Ma", "Buôn làng", "Nhà rông", "Cồng chiêng", "Thổ cẩm",
            "Lễ cúng bến nước", "Mùa rẫy", "Già làng", "Trưởng bản"
        ],
        "slang_genz": [
            "Gét gô", "Lét gô", "Khum", "Hong", "U là trời", "Ét o ét", "SOS", "Xin lũi",
            "Chằm Zn", "Trầm kẽm", "Xu cà na", "Trmúa hmề", "Chúa hề", "Cơm chó",
            "Pha ke", "Real", "Trap boy", "Trap girl", "Green flag", "Red flag",
            "Flex", "Check var", "Sao kê", "Thao túng tâm lý", "Mận", "Keo lỳ",
            "Mãi mận", "Mãi đỉnh", "Out trình", "Hảo hán", "Quay xe", "Tới công chuyện"
        ],
        "slang_underworld": [
            "Giang hồ", "Xã hội đen", "Đàn anh", "Đại ca", "Tiểu đệ", "Bảo kê",
            "Thu họ", "Vay nóng", "Bốc bát họ", "Tín dụng đen", "Đòi nợ thuê",
            "Đá", "Ke", "Kẹo", "Cỏ", "Bóng", "Bánh", "Tép", "Chấm", 
            "Hàng nóng", "Hàng lạnh", "Hoa cải", "Chó lửa", "Xế nổ", "Quái xế",
            "Đua xe", "Bão đêm", "Cắn thuốc", "Bay lắc", "Xả đồ",
            "Nhập kho", "Bóc lịch", "Áo juve", "Dựa cột", "Tiêm thuốc"
        ]
    }

    for category, terms in specialized_vocab.items():
        full_vocab.update(terms)
        categories[category] = terms
        logger.info(f"Added {len(terms)} terms for category: {category}")

    # 4. Create Structure (Optimized)
    # Convert set back to sorted list for JSON
    sorted_vocab = sorted(list(full_vocab))
    
    output_data = {
        "metadata": {
            "total_count": len(sorted_vocab),
            "sources": list(SOURCES.keys()) + ["Cherry Custom", "Specialized Security Lexicon"],
            "generated_at": "2026-01-08"
        },
        "dictionary": sorted_vocab, 
        "categories": categories
    }
    
    # 5. Save
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Successfully built vocabulary artifact: {OUTPUT_FILE}")
        logger.info(f"   Total words: {len(sorted_vocab)}")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")

if __name__ == "__main__":
    build_vocabulary()
