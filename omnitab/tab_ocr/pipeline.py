"""TabOCRPipeline - Main pipeline for TAB OCR processing"""

from pathlib import Path
from typing import Union, Optional, List
from dataclasses import dataclass

from .models import TabSong, TabMeasure, TabChord, TabNote
from .preprocessor import ImageLoader, RegionDetector, LineDetector
from .recognizer import DigitOCR, SymbolOCR, PositionMapper
from .parser import ChordGrouper, MeasureDetector, TimingAnalyzer


@dataclass
class PipelineResult:
    """Result of TAB OCR pipeline"""
    success: bool
    song: Optional[TabSong]
    errors: List[str]
    warnings: List[str]
    stats: dict


class TabOCRPipeline:
    """
    Complete pipeline for recognizing guitar TAB from images.
    
    Pipeline stages:
    1. Image loading (PDF/image → page images)
    2. TAB region detection (find 6-line TAB areas)
    3. Line detection (precise line positions)
    4. Digit OCR (recognize fret numbers)
    5. Symbol OCR (recognize techniques)
    6. Position mapping (map to string/fret)
    7. Chord grouping (group simultaneous notes)
    8. Measure detection (find bar lines)
    9. Timing analysis (infer rhythm)
    10. Output generation (TabSong)
    """
    
    def __init__(self,
                 use_gpu: bool = False,
                 dpi: int = 300,
                 default_tempo: int = 120):
        """
        Initialize pipeline.
        
        Args:
            use_gpu: Use GPU for OCR
            dpi: DPI for PDF rendering
            default_tempo: Default tempo if not detected
        """
        self.use_gpu = use_gpu
        self.dpi = dpi
        self.default_tempo = default_tempo
        
        # Initialize components
        self.image_loader = ImageLoader(dpi=dpi)
        self.region_detector = RegionDetector()
        self.line_detector = LineDetector()
        self.digit_ocr = DigitOCR(use_gpu=use_gpu)
        self.symbol_ocr = SymbolOCR(use_gpu=use_gpu)
        self.position_mapper = PositionMapper()
        self.chord_grouper = ChordGrouper()
        self.measure_detector = MeasureDetector()
        self.timing_analyzer = TimingAnalyzer()
    
    def process(self,
                input_path: Union[str, Path],
                title: Optional[str] = None,
                artist: Optional[str] = None,
                tuning: Optional[List[str]] = None,
                capo: int = 0,
                tempo: Optional[int] = None) -> PipelineResult:
        """
        Process a PDF or image file to extract TAB.
        
        Args:
            input_path: Path to PDF or image file
            title: Song title (auto-detected if not provided)
            artist: Artist name
            tuning: Guitar tuning (default: standard)
            capo: Capo position (default: 0)
            tempo: Tempo in BPM (auto-detected if not provided)
            
        Returns:
            PipelineResult with TabSong and processing info
        """
        input_path = Path(input_path)
        errors = []
        warnings = []
        
        # Initialize song
        song = TabSong(
            title=title or input_path.stem,
            artist=artist or "",
            capo=capo,
            tempo=tempo or self.default_tempo,
            source_file=str(input_path)
        )
        
        if tuning:
            song.set_tuning(tuning)
        
        try:
            # Stage 1: Load images
            print(f"[1/9] Loading images from {input_path}...")
            pages = self.image_loader.load(input_path)
            song.page_count = len(pages)
            print(f"       Loaded {len(pages)} page(s)")
            
            all_measures = []
            
            for page in pages:
                print(f"\n[Processing page {page.page_number}]")
                
                # Stage 2: Detect TAB regions
                print(f"[2/9] Detecting TAB regions...")
                regions = self.region_detector.detect_with_preprocessing(
                    page.image, page.page_number
                )
                print(f"       Found {len(regions)} TAB region(s)")
                
                if not regions:
                    warnings.append(f"No TAB regions found on page {page.page_number}")
                    continue
                
                for region in regions:
                    print(f"\n  [Region {region.region_number}]")
                    
                    # Stage 3: Detect lines
                    print(f"  [3/9] Detecting TAB lines...")
                    tab_lines = self.line_detector.detect(
                        region.image, 
                        region.line_positions
                    )
                    print(f"         {len(tab_lines.positions)} lines, "
                          f"spacing={tab_lines.spacing:.1f}px, "
                          f"confidence={tab_lines.confidence:.2f}")
                    
                    if not tab_lines.is_valid:
                        warnings.append(
                            f"Invalid line detection in page {page.page_number}, "
                            f"region {region.region_number}"
                        )
                        continue
                    
                    # Stage 4: Digit OCR
                    print(f"  [4/9] Recognizing fret numbers...")
                    digits = self.digit_ocr.recognize(
                        region.image, 
                        tab_lines.positions
                    )
                    print(f"         Found {len(digits)} digit(s)")
                    
                    if not digits:
                        warnings.append(
                            f"No digits found in page {page.page_number}, "
                            f"region {region.region_number}"
                        )
                        continue
                    
                    # Stage 5: Symbol OCR
                    print(f"  [5/9] Recognizing technique symbols...")
                    symbols = self.symbol_ocr.recognize(region.image)
                    print(f"         Found {len(symbols)} symbol(s)")
                    
                    # Stage 6: Position mapping
                    print(f"  [6/9] Mapping to TAB positions...")
                    mapped_notes = self.position_mapper.map_digits_to_notes(
                        digits, symbols, tab_lines
                    )
                    print(f"         Mapped {len(mapped_notes)} note(s)")
                    
                    # Stage 7: Chord grouping
                    print(f"  [7/9] Grouping chords...")
                    chords = self.chord_grouper.group(mapped_notes)
                    print(f"         Created {len(chords)} chord(s)")
                    
                    # Stage 8: Measure detection
                    print(f"  [8/9] Detecting measures...")
                    bars = self.measure_detector.detect_bars(
                        region.image, 
                        tab_lines.positions
                    )
                    measures = self.measure_detector.split_into_measures(
                        chords, bars
                    )
                    print(f"         Found {len(bars)} bar line(s), "
                          f"{len(measures)} measure(s)")
                    
                    # Stage 9: Timing analysis
                    print(f"  [9/9] Analyzing timing...")
                    measures = self.timing_analyzer.analyze_measures(
                        measures, method='proportional'
                    )
                    
                    all_measures.extend(measures)
            
            # Renumber measures
            for i, measure in enumerate(all_measures):
                measure.number = i + 1
                song.add_measure(measure)
            
            # Calculate stats
            stats = song.get_stats()
            
            print(f"\n[Complete!]")
            print(f"  Measures: {stats['measures']}")
            print(f"  Chords: {stats['chords']}")
            print(f"  Notes: {stats['notes']}")
            
            return PipelineResult(
                success=True,
                song=song,
                errors=errors,
                warnings=warnings,
                stats=stats
            )
            
        except Exception as e:
            errors.append(f"Pipeline error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return PipelineResult(
                success=False,
                song=None,
                errors=errors,
                warnings=warnings,
                stats={}
            )
    
    def process_image(self,
                      image,  # np.ndarray
                      page_number: int = 1) -> List[TabMeasure]:
        """
        Process a single image (already loaded).
        
        Args:
            image: Image array (BGR)
            page_number: Page number for reference
            
        Returns:
            List of TabMeasure objects
        """
        all_measures = []
        
        # Detect TAB regions
        regions = self.region_detector.detect_with_preprocessing(
            image, page_number
        )
        
        for region in regions:
            # Detect lines
            tab_lines = self.line_detector.detect(
                region.image, region.line_positions
            )
            
            if not tab_lines.is_valid:
                continue
            
            # OCR
            digits = self.digit_ocr.recognize(region.image, tab_lines.positions)
            symbols = self.symbol_ocr.recognize(region.image)
            
            if not digits:
                continue
            
            # Map and group
            mapped_notes = self.position_mapper.map_digits_to_notes(
                digits, symbols, tab_lines
            )
            chords = self.chord_grouper.group(mapped_notes)
            
            # Detect measures
            bars = self.measure_detector.detect_bars(
                region.image, tab_lines.positions
            )
            measures = self.measure_detector.split_into_measures(chords, bars)
            
            # Analyze timing
            measures = self.timing_analyzer.analyze_measures(
                measures, method='proportional'
            )
            
            all_measures.extend(measures)
        
        return all_measures
