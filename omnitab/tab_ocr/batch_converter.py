"""
Batch Converter - Convert multiple TAB images and merge into single GP5

Handles:
- Multiple page conversion
- Merging into single GP5 file
- Progress tracking
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import guitarpro as gp

from .complete_converter import CompleteConverter
from .recognizer.header_detector import HeaderDetector

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """Result from a single page conversion"""
    page_num: int
    measures: int = 0
    notes: int = 0
    success: bool = False
    error: Optional[str] = None
    gp5_path: Optional[str] = None


@dataclass  
class BatchResult:
    """Result from batch conversion"""
    total_pages: int = 0
    successful_pages: int = 0
    total_measures: int = 0
    total_notes: int = 0
    merged_file: Optional[str] = None
    page_results: List[PageResult] = field(default_factory=list)
    tuning: List[str] = field(default_factory=list)
    capo: int = 0


class BatchConverter:
    """Convert multiple TAB images and merge into single GP5"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.converter = CompleteConverter(gemini_api_key=gemini_api_key)
        self.header_detector = HeaderDetector()
    
    def convert_batch(
        self,
        image_paths: List[str],
        output_path: str,
        title: str = "OmniTab Batch",
        use_gemini: bool = True,
        merge: bool = True
    ) -> BatchResult:
        """
        Convert multiple images and optionally merge into single GP5
        
        Args:
            image_paths: List of image file paths
            output_path: Output GP5 file path
            title: Song title
            use_gemini: Use Gemini for rhythm analysis
            merge: Merge all pages into single GP5
            
        Returns:
            BatchResult with conversion details
        """
        result = BatchResult(total_pages=len(image_paths))
        page_songs = []
        
        # Detect tuning/capo from first page
        if image_paths:
            try:
                header_info = self.header_detector.detect(image_paths[0])
                result.tuning = header_info.tuning
                result.capo = header_info.capo
                logger.info(f"[Batch] Detected tuning: {result.tuning}, capo: {result.capo}")
            except Exception as e:
                logger.warning(f"[Batch] Header detection failed: {e}")
        
        # Convert each page
        for i, img_path in enumerate(image_paths):
            page_num = i + 1
            logger.info(f"[Batch] Processing page {page_num}/{len(image_paths)}: {img_path}")
            
            try:
                # Output path for individual page
                page_output = str(Path(output_path).parent / f"page_{page_num}.gp5")
                
                # Convert
                conv_result = self.converter.convert(
                    image_path=img_path,
                    output_path=page_output,
                    title=f"{title} - Page {page_num}",
                    use_gemini=use_gemini
                )
                
                page_result = PageResult(
                    page_num=page_num,
                    measures=conv_result.get("measures", 0),
                    notes=conv_result.get("notes", 0),
                    success=True,
                    gp5_path=page_output
                )
                
                # Load for merging
                if merge and os.path.exists(page_output):
                    try:
                        song = gp.parse(page_output)
                        page_songs.append(song)
                    except Exception as e:
                        logger.warning(f"[Batch] Could not load page {page_num} for merging: {e}")
                
                result.successful_pages += 1
                result.total_measures += page_result.measures
                result.total_notes += page_result.notes
                
            except Exception as e:
                logger.error(f"[Batch] Page {page_num} failed: {e}")
                page_result = PageResult(
                    page_num=page_num,
                    success=False,
                    error=str(e)
                )
            
            result.page_results.append(page_result)
        
        # Merge if requested
        if merge and page_songs:
            try:
                merged_song = self._merge_songs(page_songs, title, result.tuning, result.capo)
                gp.write(merged_song, output_path)
                result.merged_file = output_path
                logger.info(f"[Batch] Merged {len(page_songs)} pages into {output_path}")
            except Exception as e:
                logger.error(f"[Batch] Merge failed: {e}")
        
        return result
    
    def _merge_songs(
        self,
        songs: List[gp.Song],
        title: str,
        tuning: List[str],
        capo: int
    ) -> gp.Song:
        """Merge multiple GP5 songs into one"""
        if not songs:
            raise ValueError("No songs to merge")
        
        # Use first song as base
        merged = songs[0]
        merged.title = title
        merged.artist = "OmniTab Batch"
        
        # Get the first track
        if not merged.tracks:
            raise ValueError("First song has no tracks")
        
        main_track = merged.tracks[0]
        
        # Append measures from other songs
        for song in songs[1:]:
            if not song.tracks:
                continue
            
            track = song.tracks[0]
            
            # Add measure headers
            for i, header in enumerate(song.measureHeaders):
                if i < len(track.measures):
                    # Clone header
                    new_header = gp.MeasureHeader()
                    new_header.number = len(merged.measureHeaders) + 1
                    new_header.start = merged.measureHeaders[-1].start + merged.measureHeaders[-1].length if merged.measureHeaders else 960
                    new_header.timeSignature = header.timeSignature
                    new_header.isRepeatOpen = header.isRepeatOpen
                    new_header.repeatClose = header.repeatClose
                    
                    merged.measureHeaders.append(new_header)
                    
                    # Clone measure
                    src_measure = track.measures[i]
                    new_measure = gp.Measure(main_track, new_header)
                    
                    # Copy voices
                    for v_idx, voice in enumerate(src_measure.voices):
                        if v_idx < len(new_measure.voices):
                            new_measure.voices[v_idx].beats = voice.beats
                    
                    main_track.measures.append(new_measure)
        
        return merged


def convert_directory(
    directory: str,
    output_path: str,
    pattern: str = "*.png",
    **kwargs
) -> BatchResult:
    """
    Convert all matching images in a directory
    
    Args:
        directory: Directory containing images
        output_path: Output GP5 file path
        pattern: Glob pattern for images
        **kwargs: Additional arguments for BatchConverter.convert_batch
    """
    from glob import glob
    
    dir_path = Path(directory)
    image_paths = sorted(glob(str(dir_path / pattern)))
    
    if not image_paths:
        raise ValueError(f"No images found matching {pattern} in {directory}")
    
    converter = BatchConverter()
    return converter.convert_batch(image_paths, output_path, **kwargs)


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m omnitab.tab_ocr.batch_converter <directory> <output.gp5> [title]")
        sys.exit(1)
    
    directory = sys.argv[1]
    output = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "OmniTab Batch"
    
    logging.basicConfig(level=logging.INFO)
    
    result = convert_directory(directory, output, title=title)
    
    print(f"\n{'='*50}")
    print(f"Batch Conversion Complete")
    print(f"{'='*50}")
    print(f"Pages: {result.successful_pages}/{result.total_pages}")
    print(f"Total measures: {result.total_measures}")
    print(f"Total notes: {result.total_notes}")
    if result.merged_file:
        print(f"Output: {result.merged_file}")
