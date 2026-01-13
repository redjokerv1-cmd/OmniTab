"""PositionMapper - Map detected elements to TAB positions"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..models import TabNote, Technique, HarmonicType
from ..preprocessor.line_detector import TabLines
from .digit_ocr import DetectedDigit
from .symbol_ocr import DetectedSymbol, SymbolType


@dataclass
class MappedNote:
    """A digit mapped to a TAB position with optional technique"""
    note: TabNote
    digit: DetectedDigit
    symbols: List[DetectedSymbol]


class PositionMapper:
    """
    Map detected digits and symbols to guitar TAB positions.
    
    Converts:
    - Digit y-position → Guitar string (1-6)
    - Digit x-position → Time position
    - Nearby symbols → Note techniques
    """
    
    def __init__(self, symbol_search_radius: float = 25):
        """
        Initialize PositionMapper.
        
        Args:
            symbol_search_radius: Radius to search for symbols near digits
        """
        self.symbol_search_radius = symbol_search_radius
    
    def map_digits_to_notes(self,
                           digits: List[DetectedDigit],
                           symbols: List[DetectedSymbol],
                           tab_lines: TabLines) -> List[MappedNote]:
        """
        Map detected digits and symbols to TabNote objects.
        
        Args:
            digits: List of detected digits (fret numbers)
            symbols: List of detected symbols (techniques)
            tab_lines: Detected TAB line positions
            
        Returns:
            List of MappedNote objects
        """
        if not tab_lines.is_valid:
            # Can't map without valid line positions
            return []
        
        mapped = []
        
        for digit in digits:
            # Map y position to string number
            string = tab_lines.get_string_for_y(digit.y)
            
            # Find nearby symbols
            nearby_symbols = self._find_nearby_symbols(
                digit, symbols, self.symbol_search_radius
            )
            
            # Create TabNote
            note = TabNote(
                string=string,
                fret=digit.value,
                x_position=digit.x,
                y_position=digit.y,
                confidence=digit.confidence
            )
            
            # Apply techniques from symbols
            self._apply_techniques(note, nearby_symbols)
            
            mapped.append(MappedNote(
                note=note,
                digit=digit,
                symbols=nearby_symbols
            ))
        
        return mapped
    
    def _find_nearby_symbols(self,
                             digit: DetectedDigit,
                             symbols: List[DetectedSymbol],
                             radius: float) -> List[DetectedSymbol]:
        """Find symbols within radius of a digit"""
        nearby = []
        
        for symbol in symbols:
            distance = ((symbol.x - digit.x) ** 2 + 
                       (symbol.y - digit.y) ** 2) ** 0.5
            
            if distance <= radius:
                nearby.append(symbol)
        
        return nearby
    
    def _apply_techniques(self, 
                          note: TabNote,
                          symbols: List[DetectedSymbol]) -> None:
        """Apply technique information from symbols to a note"""
        for symbol in symbols:
            st = symbol.symbol_type
            
            # Playing techniques
            if st == SymbolType.HAMMER_ON:
                note.technique = Technique.HAMMER_ON
                note.connected_to_next = True
            
            elif st == SymbolType.PULL_OFF:
                note.technique = Technique.PULL_OFF
                note.connected_to_next = True
            
            elif st == SymbolType.SLIDE_UP:
                note.technique = Technique.SLIDE_UP
                note.connected_to_next = True
            
            elif st == SymbolType.SLIDE_DOWN:
                note.technique = Technique.SLIDE_DOWN
                note.connected_to_next = True
            
            elif st == SymbolType.BEND:
                note.technique = Technique.BEND
            
            elif st == SymbolType.RELEASE:
                note.technique = Technique.RELEASE
            
            elif st == SymbolType.VIBRATO:
                note.technique = Technique.VIBRATO
            
            elif st == SymbolType.TAP:
                note.technique = Technique.TAP
            
            elif st == SymbolType.PALM_MUTE:
                note.technique = Technique.PALM_MUTE
            
            # Harmonics
            elif st == SymbolType.NATURAL_HARMONIC:
                note.harmonic_type = HarmonicType.NATURAL
                note.harmonic_fret = symbol.harmonic_fret
            
            elif st == SymbolType.ARTIFICIAL_HARMONIC:
                note.harmonic_type = HarmonicType.ARTIFICIAL
            
            elif st == SymbolType.PINCH_HARMONIC:
                note.harmonic_type = HarmonicType.PINCH
            
            elif st == SymbolType.TAP_HARMONIC:
                note.harmonic_type = HarmonicType.TAP
            
            # Muting
            elif st == SymbolType.MUTE:
                note.is_muted = True
                note.is_dead = True
            
            elif st == SymbolType.GHOST:
                note.is_ghost = True
    
    def estimate_string_from_relative_position(self,
                                               y: float,
                                               region_height: float,
                                               num_strings: int = 6) -> int:
        """
        Estimate string number from relative y position.
        
        Fallback when exact line positions aren't available.
        
        Args:
            y: Y position in region
            region_height: Total height of region
            num_strings: Number of strings (default 6)
            
        Returns:
            Estimated string number (1-6)
        """
        # Assume TAB takes up ~70% of region height, centered
        tab_start = region_height * 0.15
        tab_end = region_height * 0.85
        tab_height = tab_end - tab_start
        
        # Calculate relative position within TAB area
        relative_y = (y - tab_start) / tab_height
        
        # Clamp to valid range
        relative_y = max(0, min(1, relative_y))
        
        # Map to string number
        string = int(relative_y * num_strings) + 1
        return min(max(string, 1), num_strings)
    
    def group_notes_by_x_position(self,
                                  notes: List[MappedNote],
                                  x_threshold: float = 15) -> List[List[MappedNote]]:
        """
        Group notes that are at the same x position (chords).
        
        Args:
            notes: List of mapped notes
            x_threshold: Maximum x distance to consider same position
            
        Returns:
            List of note groups (each group is played simultaneously)
        """
        if not notes:
            return []
        
        # Sort by x position
        sorted_notes = sorted(notes, key=lambda n: n.note.x_position)
        
        groups = []
        current_group = [sorted_notes[0]]
        
        for note in sorted_notes[1:]:
            if note.note.x_position - current_group[-1].note.x_position < x_threshold:
                current_group.append(note)
            else:
                groups.append(current_group)
                current_group = [note]
        
        groups.append(current_group)
        return groups
