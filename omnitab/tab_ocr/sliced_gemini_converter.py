"""
Sliced Gemini Converter

Enhanced conversion pipeline:
1. Slice score into systems (lines)
2. Optionally slice systems into measures
3. Send each slice to Gemini for analysis
4. Merge results and generate GP5

This approach improves accuracy by letting Gemini focus on smaller regions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import guitarpro as gp

from .preprocessor.score_slicer import ScoreSlicer, SlicedRegion
from .gemini_only_converter import GeminiOnlyConverter

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    from .gemini_analyzer import GeminiTabAnalyzer, DURATION_TO_GP5
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


class SlicedGeminiConverter:
    """
    Convert TAB images to GP5 using sliced approach
    
    Pipeline:
    1. Slice image into systems (horizontal lines)
    2. Analyze each system with Gemini
    3. Merge all measures
    4. Generate GP5
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini not available. Install: pip install google-generativeai")
        
        self.analyzer = GeminiTabAnalyzer(api_key=api_key)
        self.slicer = ScoreSlicer()
        logger.info("[SlicedGemini] Initialized")
    
    def convert(
        self,
        image_path: str,
        output_path: str,
        title: str = "OmniTab (Sliced)",
        tempo: int = 120,
        slice_measures: bool = False
    ) -> Dict:
        """
        Convert TAB image to GP5 using sliced approach
        
        Args:
            image_path: Path to TAB image
            output_path: Output GP5 file path
            title: Song title
            tempo: Default tempo
            slice_measures: Also slice into individual measures (more API calls)
            
        Returns:
            Dict with conversion results
        """
        logger.info(f"[SlicedGemini] Converting: {image_path}")
        print("=" * 60)
        print("[SlicedGemini] Enhanced Conversion Pipeline")
        print("=" * 60)
        
        # Step 1: Slice into systems
        print("\n[1] Slicing into systems...")
        systems = self.slicer.slice_into_systems(image_path)
        print(f"    Found {len(systems)} systems")
        
        if not systems:
            raise ValueError("No systems detected in image")
        
        # Step 2: Analyze each system
        all_measures = []
        detected_tuning = None
        detected_capo = 0
        detected_tempo = tempo
        
        for i, system in enumerate(systems):
            print(f"\n[2.{i+1}] Analyzing system {i+1}/{len(systems)}...")
            
            if slice_measures:
                # Slice system into measures and analyze each
                measures = self.slicer.slice_into_measures(system.path, save=True)
                print(f"    Found {len(measures)} measures in system")
                
                for m in measures:
                    try:
                        result = self.analyzer.analyze(m.path)
                        measure_data = result.get("measures", [])
                        all_measures.extend(measure_data)
                        
                        # Get metadata from first measure
                        if detected_tuning is None:
                            detected_tuning = result.get("tuning")
                            detected_capo = result.get("capo", 0)
                            detected_tempo = result.get("tempo", tempo)
                    except Exception as e:
                        logger.warning(f"[SlicedGemini] Measure {m.index} failed: {e}")
            else:
                # Analyze whole system at once
                try:
                    result = self.analyzer.analyze(system.path)
                    measure_data = result.get("measures", [])
                    all_measures.extend(measure_data)
                    print(f"    Found {len(measure_data)} measures")
                    
                    # Get metadata from first system
                    if detected_tuning is None:
                        detected_tuning = result.get("tuning")
                        detected_capo = result.get("capo", 0)
                        detected_tempo = result.get("tempo", tempo)
                except Exception as e:
                    logger.error(f"[SlicedGemini] System {i} failed: {e}")
        
        # Use defaults if not detected
        if detected_tuning is None:
            detected_tuning = ["E", "B", "G", "D", "A", "E"]
        
        # Count notes
        total_notes = sum(
            len(beat.get("notes", []))
            for m in all_measures
            for beat in m.get("beats", [])
        )
        
        print(f"\n[3] Results:")
        print(f"    Total measures: {len(all_measures)}")
        print(f"    Total notes: {total_notes}")
        print(f"    Tuning: {detected_tuning}")
        print(f"    Capo: {detected_capo}")
        print(f"    Tempo: {detected_tempo}")
        
        # Step 3: Generate GP5
        print("\n[4] Generating GP5...")
        song = self._create_gp5(all_measures, title, detected_tempo, 
                               detected_tuning, detected_capo)
        
        gp.write(song, output_path)
        print(f"    [OK] Saved: {output_path}")
        
        # Count beats
        gp5_beats = sum(
            len(v.beats)
            for t in song.tracks
            for m in t.measures
            for v in m.voices
        )
        
        print("\n" + "=" * 60)
        print("RESULT (Sliced Gemini)")
        print("=" * 60)
        print(f"  Systems analyzed: {len(systems)}")
        print(f"  Measures: {len(all_measures)}")
        print(f"  Beats: {gp5_beats}")
        print(f"  Notes: {total_notes}")
        
        return {
            "output": output_path,
            "systems": len(systems),
            "measures": len(all_measures),
            "notes": total_notes,
            "beats": gp5_beats,
            "rhythm_source": "gemini_sliced",
            "tuning": detected_tuning,
            "capo": detected_capo,
            "tempo": detected_tempo,
            "mode": "sliced_gemini"
        }
    
    def _create_gp5(
        self,
        measures_data: List[Dict],
        title: str,
        tempo: int,
        tuning: List[str],
        capo: int
    ) -> gp.Song:
        """Create GP5 song (same as GeminiOnlyConverter)"""
        # Reuse the GeminiOnlyConverter's GP5 creation logic
        converter = GeminiOnlyConverter.__new__(GeminiOnlyConverter)
        return converter._create_gp5(measures_data, title, tempo, tuning, capo)


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m omnitab.tab_ocr.sliced_gemini_converter <image> <output.gp5>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    converter = SlicedGeminiConverter()
    result = converter.convert(sys.argv[1], sys.argv[2])
    print(f"\nDone! Notes: {result['notes']}")
