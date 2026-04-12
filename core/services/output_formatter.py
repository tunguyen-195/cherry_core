from typing import List, Dict

class OutputFormatter:
    """
    Handles formatting of transcript output for various requirements.
    """
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Convert seconds to HH:MM:SS,mmm format."""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        ms = int((s - int(s)) * 1000)
        return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

    @staticmethod
    def format_subtitle_style(aligned_segments: List[Dict]) -> str:
        """
        Formats aligned segments into a subtitle-like structure with timestamps.
        Format:
        00:00:00,100 --> 00:00:20,419 [Speaker X]
        Text content...
        """
        blocks = []
        current_block = None
        
        for seg in aligned_segments:
            spk = seg["speaker"]
            if current_block and current_block["speaker"] == spk:
                current_block["text"].append(seg["text"])
                current_block["end"] = seg["end"]
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = {
                    "speaker": spk,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": [seg["text"]]
                }
        if current_block:
            blocks.append(current_block)
            
        output_lines = []
        for b in blocks:
            start_str = OutputFormatter.format_time(b["start"])
            end_str = OutputFormatter.format_time(b["end"])
            full_text = " ".join(b["text"])
            
            # Format: 00:00:51,520 --> 00:01:11,750 [Speaker 0]
            # Convert internal "SPEAKER_01" -> "Speaker 1" if possible, or just Title Case
            spk_label = b['speaker']
            if spk_label.upper().startswith("SPEAKER_"):
                try:
                    num = int(spk_label.split("_")[1])
                    spk_label = f"Speaker {num}"
                except:
                    spk_label = spk_label.title().replace("_", " ")
            else:
                 spk_label = spk_label.title()
            
            output_lines.append(f"{start_str} --> {end_str} [{spk_label}]")
            output_lines.append(f"{full_text}\n")
            
        return "\n".join(output_lines)
