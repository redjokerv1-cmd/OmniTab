"""TimingAnalyzer - Analyze rhythm and timing from note positions"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from ..models import TabChord, TabMeasure, Duration


@dataclass
class TimingInfo:
    """Timing information for a chord"""
    duration: Duration
    confidence: float
    beat_position: float  # Position within measure (0.0 - 4.0 for 4/4)


class TimingAnalyzer:
    """
    Analyze rhythm and timing from spatial positions of notes.
    
    Methods:
    1. Uniform spacing - assume equal duration notes
    2. Proportional spacing - infer duration from x-distances
    3. Beat grid - snap to regular beat positions
    """
    
    def __init__(self, 
                 time_signature: Tuple[int, int] = (4, 4),
                 tempo: int = 120):
        """
        Initialize TimingAnalyzer.
        
        Args:
            time_signature: (beats, beat_value)
            tempo: BPM
        """
        self.time_signature = time_signature
        self.tempo = tempo
    
    def analyze_uniform(self, 
                        chords: List[TabChord],
                        duration: Duration = Duration.EIGHTH) -> List[TabChord]:
        """
        Apply uniform duration to all chords.
        
        Simple approach - assumes all notes have same duration.
        
        Args:
            chords: List of TabChord objects
            duration: Duration to apply to all
            
        Returns:
            Updated chords with duration set
        """
        for chord in chords:
            chord.duration = duration
        return chords
    
    def analyze_proportional(self, 
                             chords: List[TabChord],
                             measure_width: Optional[float] = None) -> List[TabChord]:
        """
        Infer duration from relative spacing between chords.
        
        Larger gaps = longer notes, smaller gaps = shorter notes.
        
        Args:
            chords: List of TabChord objects
            measure_width: Optional width of one measure in pixels
            
        Returns:
            Updated chords with duration set
        """
        if len(chords) < 2:
            return self.analyze_uniform(chords)
        
        # Calculate gaps between chords
        gaps = []
        for i in range(len(chords) - 1):
            gap = chords[i + 1].x_position - chords[i].x_position
            gaps.append(gap)
        
        if not gaps:
            return self.analyze_uniform(chords)
        
        # Determine duration thresholds
        avg_gap = np.mean(gaps)
        
        # Map gaps to durations
        for i, chord in enumerate(chords):
            if i < len(gaps):
                gap = gaps[i]
                duration = self._gap_to_duration(gap, avg_gap)
            else:
                # Last chord - use previous duration or default
                duration = chords[i - 1].duration if i > 0 else Duration.QUARTER
            
            chord.duration = duration
        
        return chords
    
    def analyze_beat_grid(self,
                          chords: List[TabChord],
                          measure_x_start: float,
                          measure_x_end: float,
                          subdivisions: int = 16) -> List[TabChord]:
        """
        Snap chords to a beat grid.
        
        Divides measure into subdivisions and assigns chords
        to nearest grid position.
        
        Args:
            chords: List of TabChord objects
            measure_x_start: X position of measure start
            measure_x_end: X position of measure end
            subdivisions: Number of subdivisions (16 = sixteenth notes)
            
        Returns:
            Updated chords with duration set
        """
        if not chords:
            return chords
        
        measure_width = measure_x_end - measure_x_start
        if measure_width <= 0:
            return self.analyze_uniform(chords)
        
        grid_width = measure_width / subdivisions
        beats_per_measure = self.time_signature[0]
        subdivisions_per_beat = subdivisions // beats_per_measure
        
        # Snap each chord to grid
        for chord in chords:
            # Calculate position within measure
            relative_x = chord.x_position - measure_x_start
            grid_position = round(relative_x / grid_width)
            grid_position = max(0, min(subdivisions - 1, grid_position))
            
            # Determine duration based on gap to next grid position
            # For now, use default based on common subdivision
            if subdivisions_per_beat == 4:  # 16ths
                chord.duration = Duration.SIXTEENTH
            elif subdivisions_per_beat == 2:  # 8ths
                chord.duration = Duration.EIGHTH
            else:
                chord.duration = Duration.QUARTER
        
        return chords
    
    def analyze_measures(self,
                        measures: List[TabMeasure],
                        method: str = 'proportional') -> List[TabMeasure]:
        """
        Analyze timing for all measures.
        
        Args:
            measures: List of TabMeasure objects
            method: 'uniform', 'proportional', or 'grid'
            
        Returns:
            Updated measures with timing info
        """
        for measure in measures:
            if not measure.chords:
                continue
            
            if method == 'uniform':
                self.analyze_uniform(measure.chords)
            elif method == 'proportional':
                self.analyze_proportional(measure.chords)
            elif method == 'grid':
                if measure.chords:
                    x_start = measure.x_start or measure.chords[0].x_position
                    x_end = measure.x_end or measure.chords[-1].x_position + 20
                    self.analyze_beat_grid(measure.chords, x_start, x_end)
            else:
                self.analyze_uniform(measure.chords)
        
        return measures
    
    def _gap_to_duration(self, gap: float, avg_gap: float) -> Duration:
        """Map a gap size to a duration"""
        ratio = gap / avg_gap if avg_gap > 0 else 1.0
        
        if ratio < 0.4:
            return Duration.SIXTEENTH
        elif ratio < 0.7:
            return Duration.EIGHTH
        elif ratio < 1.3:
            return Duration.QUARTER
        elif ratio < 2.0:
            return Duration.HALF
        else:
            return Duration.WHOLE
    
    def calculate_measure_duration(self, measure: TabMeasure) -> float:
        """
        Calculate total duration of notes in a measure.
        
        Returns duration in beats.
        """
        total = 0.0
        beat_value = self.time_signature[1]  # 4 for quarter note
        
        for chord in measure.chords:
            # Convert duration enum to beats
            note_value = chord.duration.value
            # Duration.QUARTER = 4, which is 1 beat in 4/4
            beats = beat_value / note_value
            total += beats
        
        return total
    
    def validate_measure_timing(self, 
                                measure: TabMeasure,
                                tolerance: float = 0.1) -> bool:
        """
        Check if measure duration matches time signature.
        
        Args:
            measure: TabMeasure to validate
            tolerance: Allowed deviation from expected beats
            
        Returns:
            True if timing is valid
        """
        expected_beats = self.time_signature[0]
        actual_beats = self.calculate_measure_duration(measure)
        
        return abs(actual_beats - expected_beats) <= tolerance
    
    def adjust_timing_to_fit(self, measure: TabMeasure) -> TabMeasure:
        """
        Adjust note durations to fit the time signature.
        
        If measure is too long/short, scale durations proportionally.
        """
        if not measure.chords:
            return measure
        
        expected_beats = self.time_signature[0]
        actual_beats = self.calculate_measure_duration(measure)
        
        if abs(actual_beats - expected_beats) < 0.1:
            return measure  # Already correct
        
        # Scale factor
        scale = expected_beats / actual_beats if actual_beats > 0 else 1.0
        
        # For now, just pick a duration that fits
        num_chords = len(measure.chords)
        beats_per_chord = expected_beats / num_chords
        
        # Find closest duration
        if beats_per_chord >= 2:
            duration = Duration.HALF
        elif beats_per_chord >= 1:
            duration = Duration.QUARTER
        elif beats_per_chord >= 0.5:
            duration = Duration.EIGHTH
        else:
            duration = Duration.SIXTEENTH
        
        for chord in measure.chords:
            chord.duration = duration
        
        return measure
