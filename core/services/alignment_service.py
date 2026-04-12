"""
Alignment Service (Domain Service).
Aligns Whisper Word-Level Timestamps with Diarization Segments.
Optimized with IntervalTree for O(log n) speaker lookup.
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import intervaltree for optimized lookup
try:
    from intervaltree import IntervalTree
    INTERVALTREE_AVAILABLE = True
except ImportError:
    INTERVALTREE_AVAILABLE = False
    logger.warning("intervaltree not installed. Using linear search (slower for large files).")


class AlignmentService:
    """
    Core Alignment Logic.
    Uses IntervalTree for O(log n) speaker segment lookup when available.
    Falls back to linear search if intervaltree not installed.
    """
    
    @staticmethod
    def align_words(transcript_segments: List[Dict], speaker_segments: List[Any]) -> List[Dict]:
        """
        Align transcript words to speakers.
        
        Args:
            transcript_segments: Whisper segments (must contain 'words' list if available)
            speaker_segments: List[SpeakerSegment] objects
            
        Returns:
            List of aligned segments with precise speaker labels.
        """
        aligned_result = []
        
        # Flatten all words from transcript
        all_words = []
        for t_seg in transcript_segments:
            if "words" in t_seg:
                all_words.extend(t_seg["words"])
            else:
                all_words.append({
                    "word": t_seg["text"],
                    "start": t_seg["start"],
                    "end": t_seg["end"]
                })
        
        if not all_words:
            return aligned_result
        
        # Build speaker lookup structure
        if INTERVALTREE_AVAILABLE:
            speaker_lookup = AlignmentService._build_interval_tree(speaker_segments)
            find_speaker = lambda t: AlignmentService._find_speaker_intervaltree(t, speaker_lookup, speaker_segments)
        else:
            find_speaker = lambda t: AlignmentService._find_speaker_linear(t, speaker_segments)
        
        # Align each word
        current_speaker_block = {"speaker": None, "text": [], "start": 0, "end": 0}
        
        for word in all_words:
            w_start = word["start"]
            w_end = word["end"]
            w_mid = (w_start + w_end) / 2
            
            assigned_speaker = find_speaker(w_mid)
            
            # If changed speaker, push block
            if assigned_speaker != current_speaker_block["speaker"]:
                if current_speaker_block["speaker"] is not None:
                    aligned_result.append({
                        "speaker": current_speaker_block["speaker"],
                        "text": " ".join(current_speaker_block["text"]),
                        "start": current_speaker_block["start"],
                        "end": w_start 
                    })
                
                current_speaker_block = {
                    "speaker": assigned_speaker,
                    "text": [word["word"].strip()],
                    "start": w_start,
                    "end": w_end
                }
            else:
                current_speaker_block["text"].append(word["word"].strip())
                current_speaker_block["end"] = w_end
                
        # Push last block
        if current_speaker_block["speaker"] is not None:
            aligned_result.append({
                "speaker": current_speaker_block["speaker"],
                "text": " ".join(current_speaker_block["text"]),
                "start": current_speaker_block["start"],
                "end": current_speaker_block["end"]
            })
            
        return aligned_result
    
    @staticmethod
    def _build_interval_tree(speaker_segments: List[Any]) -> "IntervalTree":
        """Build IntervalTree from speaker segments for O(log n) lookup."""
        tree = IntervalTree()
        for seg in speaker_segments:
            # IntervalTree uses half-open intervals [start, end)
            # Add small epsilon to make it closed [start, end]
            tree[seg.start_time:seg.end_time + 0.001] = seg.speaker_id
        return tree
    
    @staticmethod
    def _find_speaker_intervaltree(time: float, tree: "IntervalTree", speaker_segments: List[Any]) -> str:
        """Find speaker at given time using IntervalTree (O(log n))."""
        overlaps = tree[time]
        
        if overlaps:
            # Return first match
            return list(overlaps)[0].data
        
        # Fallback: find nearest speaker
        return AlignmentService._find_nearest_speaker(time, speaker_segments)
    
    @staticmethod
    def _find_speaker_linear(time: float, speaker_segments: List[Any]) -> str:
        """Find speaker at given time using linear search (O(n))."""
        nearest_speaker = None
        nearest_dist = float('inf')
        
        for seg in speaker_segments:
            if seg.start_time <= time <= seg.end_time:
                return seg.speaker_id
            
            dist = min(abs(time - seg.start_time), abs(time - seg.end_time))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_speaker = seg.speaker_id
        
        # Fallback to nearest within 2 seconds
        if nearest_speaker and nearest_dist < 2.0:
            return nearest_speaker
        
        return "UNKNOWN"
    
    @staticmethod
    def _find_nearest_speaker(time: float, speaker_segments: List[Any]) -> str:
        """Find nearest speaker when no direct overlap (for IntervalTree fallback)."""
        nearest_speaker = None
        nearest_dist = float('inf')
        
        for seg in speaker_segments:
            dist = min(abs(time - seg.start_time), abs(time - seg.end_time))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_speaker = seg.speaker_id
        
        if nearest_speaker and nearest_dist < 2.0:
            return nearest_speaker
        
        return "UNKNOWN"

