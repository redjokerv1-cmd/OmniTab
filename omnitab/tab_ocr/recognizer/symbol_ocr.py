"""SymbolOCR - Recognize technique symbols (H, P, /, etc.) in TAB"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import re
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import easyocr
except ImportError:
    easyocr = None


class SymbolType(Enum):
    """Types of symbols in guitar TAB"""
    HAMMER_ON = "h"
    PULL_OFF = "p"
    SLIDE_UP = "/"
    SLIDE_DOWN = "\\"
    BEND = "b"
    RELEASE = "r"
    VIBRATO = "~"
    TAP = "t"
    NATURAL_HARMONIC = "nh"     # <12>
    ARTIFICIAL_HARMONIC = "ah"  # AH
    PINCH_HARMONIC = "ph"
    TAP_HARMONIC = "th"
    PALM_MUTE = "pm"
    LET_RING = "lr"
    MUTE = "x"
    GHOST = "ghost"  # (number)
    STRUM_DOWN = "down"
    STRUM_UP = "up"
    UNKNOWN = "unknown"


@dataclass
class DetectedSymbol:
    """A detected technique symbol"""
    symbol_type: SymbolType
    x: float
    y: float
    width: float
    height: float
    confidence: float
    raw_text: str
    
    # For harmonics: the fret number
    harmonic_fret: Optional[int] = None


class SymbolOCR:
    """
    Recognize guitar technique symbols in TAB images.
    
    Recognizes:
    - Letters: H, P, b, r, t, x, AH, PH, TH
    - Special: /, \\, ~
    - Harmonics: <5>, <7>, <12>
    - Arrows: ↓, ↑
    """
    
    # Symbol patterns
    SYMBOL_PATTERNS = {
        r'^[Hh]$': SymbolType.HAMMER_ON,
        r'^[Pp]$': SymbolType.PULL_OFF,
        r'^[/]$': SymbolType.SLIDE_UP,
        r'^[\\]$': SymbolType.SLIDE_DOWN,
        r'^[Bb]$': SymbolType.BEND,
        r'^[Rr]$': SymbolType.RELEASE,
        r'^[~]$': SymbolType.VIBRATO,
        r'^[Tt]$': SymbolType.TAP,
        r'^[Xx]$': SymbolType.MUTE,
        r'^[Aa][Hh]$': SymbolType.ARTIFICIAL_HARMONIC,
        r'^[Pp][Hh]$': SymbolType.PINCH_HARMONIC,
        r'^[Tt][Hh]$': SymbolType.TAP_HARMONIC,
        r'^[Pp][Mm]$': SymbolType.PALM_MUTE,
        r'^<\d+>$': SymbolType.NATURAL_HARMONIC,
    }
    
    def __init__(self, 
                 use_gpu: bool = False,
                 min_confidence: float = 0.4):
        """
        Initialize SymbolOCR.
        
        Args:
            use_gpu: Use GPU for OCR
            min_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_confidence
        self.use_gpu = use_gpu
        self._reader = None
        
        if cv2 is None:
            raise ImportError("OpenCV required. Install: pip install opencv-python")
    
    @property
    def reader(self):
        """Lazy load EasyOCR reader"""
        if self._reader is None:
            if easyocr is None:
                raise ImportError("EasyOCR required. Install: pip install easyocr")
            self._reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
        return self._reader
    
    def recognize(self, image: np.ndarray) -> List[DetectedSymbol]:
        """
        Recognize technique symbols in a TAB image.
        
        Args:
            image: TAB region image (BGR or grayscale)
            
        Returns:
            List of DetectedSymbol objects
        """
        # Run OCR with letter allowlist
        results = self.reader.readtext(
            image,
            allowlist='HhPpBbRrTtXxAa<>0123456789/\\~↓↑',
            paragraph=False,
            detail=1,
            min_size=5
        )
        
        symbols = []
        
        for result in results:
            bbox, text, confidence = result
            
            if confidence < self.min_confidence:
                continue
            
            # Identify symbol type
            symbol_type, harmonic_fret = self._identify_symbol(text)
            
            if symbol_type == SymbolType.UNKNOWN:
                continue
            
            # Calculate position
            x_center = (bbox[0][0] + bbox[2][0]) / 2
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            width = bbox[2][0] - bbox[0][0]
            height = bbox[2][1] - bbox[0][1]
            
            symbols.append(DetectedSymbol(
                symbol_type=symbol_type,
                x=x_center,
                y=y_center,
                width=width,
                height=height,
                confidence=confidence,
                raw_text=text,
                harmonic_fret=harmonic_fret
            ))
        
        # Also detect arrow symbols using template matching
        arrows = self._detect_arrows(image)
        symbols.extend(arrows)
        
        return symbols
    
    def _identify_symbol(self, text: str) -> Tuple[SymbolType, Optional[int]]:
        """
        Identify symbol type from OCR text.
        
        Returns:
            (SymbolType, harmonic_fret or None)
        """
        text = text.strip()
        
        # Check each pattern
        for pattern, symbol_type in self.SYMBOL_PATTERNS.items():
            if re.match(pattern, text):
                # Extract harmonic fret if applicable
                if symbol_type == SymbolType.NATURAL_HARMONIC:
                    match = re.search(r'<(\d+)>', text)
                    if match:
                        fret = int(match.group(1))
                        return symbol_type, fret
                
                return symbol_type, None
        
        return SymbolType.UNKNOWN, None
    
    def _detect_arrows(self, image: np.ndarray) -> List[DetectedSymbol]:
        """Detect strum direction arrows using shape analysis"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        arrows = []
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Arrow-like shape: taller than wide, small area
            if h > w * 1.5 and 10 < h < 50 and w < 30:
                # Check if it's triangular (arrow shape)
                hull = cv2.convexHull(contour)
                if len(hull) >= 3:
                    # Simplified arrow detection
                    # Check center of mass position
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cy = int(M['m01'] / M['m00'])
                        
                        # If center is in upper half, it's pointing down
                        relative_cy = (cy - y) / h
                        
                        if relative_cy < 0.4:
                            arrow_type = SymbolType.STRUM_DOWN
                        elif relative_cy > 0.6:
                            arrow_type = SymbolType.STRUM_UP
                        else:
                            continue
                        
                        arrows.append(DetectedSymbol(
                            symbol_type=arrow_type,
                            x=x + w/2,
                            y=y + h/2,
                            width=w,
                            height=h,
                            confidence=0.7,
                            raw_text="↓" if arrow_type == SymbolType.STRUM_DOWN else "↑"
                        ))
        
        return arrows
    
    def find_symbols_near_digit(self,
                                symbols: List[DetectedSymbol],
                                digit_x: float,
                                digit_y: float,
                                max_distance: float = 30) -> List[DetectedSymbol]:
        """
        Find symbols near a detected digit.
        
        Args:
            symbols: List of detected symbols
            digit_x: X position of digit
            digit_y: Y position of digit
            max_distance: Maximum distance to consider "near"
            
        Returns:
            List of nearby symbols
        """
        nearby = []
        
        for symbol in symbols:
            distance = ((symbol.x - digit_x) ** 2 + 
                       (symbol.y - digit_y) ** 2) ** 0.5
            
            if distance <= max_distance:
                nearby.append(symbol)
        
        return nearby
