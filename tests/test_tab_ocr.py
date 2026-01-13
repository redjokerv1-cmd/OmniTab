"""Tests for TAB OCR system"""

import pytest
from pathlib import Path
import numpy as np

# Test data models
from omnitab.tab_ocr.models import TabNote, TabChord, TabMeasure, TabSong
from omnitab.tab_ocr.models.tab_note import Technique, HarmonicType
from omnitab.tab_ocr.models.tab_chord import Duration, StrumDirection


class TestTabNote:
    """Test TabNote model"""
    
    def test_create_basic_note(self):
        """Test creating a basic note"""
        note = TabNote(string=1, fret=5)
        assert note.string == 1
        assert note.fret == 5
        assert note.technique == Technique.NONE
        assert note.harmonic_type == HarmonicType.NONE
    
    def test_note_with_technique(self):
        """Test note with technique"""
        note = TabNote(
            string=3, 
            fret=7,
            technique=Technique.HAMMER_ON
        )
        assert note.technique == Technique.HAMMER_ON
    
    def test_note_with_harmonic(self):
        """Test note with harmonic"""
        note = TabNote(
            string=1,
            fret=12,
            harmonic_type=HarmonicType.NATURAL,
            harmonic_fret=12
        )
        assert note.harmonic_type == HarmonicType.NATURAL
        assert note.harmonic_fret == 12
    
    def test_invalid_string(self):
        """Test invalid string number"""
        with pytest.raises(ValueError):
            TabNote(string=0, fret=5)
        with pytest.raises(ValueError):
            TabNote(string=7, fret=5)
    
    def test_invalid_fret(self):
        """Test invalid fret number"""
        with pytest.raises(ValueError):
            TabNote(string=1, fret=-1)
        with pytest.raises(ValueError):
            TabNote(string=1, fret=25)
    
    def test_to_dict(self):
        """Test conversion to dict"""
        note = TabNote(
            string=2,
            fret=3,
            technique=Technique.SLIDE_UP,
            is_muted=True
        )
        d = note.to_dict()
        assert d['string'] == 2
        assert d['fret'] == 3
        assert d['technique'] == '/'
        assert d['is_muted'] == True


class TestTabChord:
    """Test TabChord model"""
    
    def test_single_note_chord(self):
        """Test chord with single note"""
        note = TabNote(string=1, fret=5)
        chord = TabChord(notes=[note])
        
        assert chord.is_single_note
        assert not chord.is_chord
        assert len(chord.notes) == 1
    
    def test_multi_note_chord(self):
        """Test chord with multiple notes"""
        notes = [
            TabNote(string=1, fret=0),
            TabNote(string=2, fret=1),
            TabNote(string=3, fret=0),
        ]
        chord = TabChord(notes=notes)
        
        assert chord.is_chord
        assert not chord.is_single_note
        assert len(chord.notes) == 3
    
    def test_rest(self):
        """Test rest chord"""
        chord = TabChord(is_rest=True, duration=Duration.QUARTER)
        
        assert chord.is_rest
        assert chord.duration == Duration.QUARTER
    
    def test_add_note(self):
        """Test adding notes to chord"""
        chord = TabChord()
        chord.add_note(TabNote(string=1, fret=5, x_position=100))
        chord.add_note(TabNote(string=2, fret=5, x_position=102))
        
        assert len(chord.notes) == 2
        assert chord.x_position == 101  # Average


class TestTabMeasure:
    """Test TabMeasure model"""
    
    def test_create_measure(self):
        """Test creating a measure"""
        measure = TabMeasure(number=1)
        assert measure.number == 1
        assert measure.time_signature == (4, 4)
        assert len(measure.chords) == 0
    
    def test_add_chords(self):
        """Test adding chords to measure"""
        measure = TabMeasure(number=1)
        
        chord1 = TabChord(
            notes=[TabNote(string=1, fret=0)],
            x_position=10
        )
        chord2 = TabChord(
            notes=[TabNote(string=2, fret=3)],
            x_position=20
        )
        
        measure.add_chord(chord1)
        measure.add_chord(chord2)
        
        assert measure.chord_count == 2
        
        sorted_chords = measure.get_chords_sorted()
        assert sorted_chords[0].x_position < sorted_chords[1].x_position


class TestTabSong:
    """Test TabSong model"""
    
    def test_create_song(self):
        """Test creating a song"""
        song = TabSong(
            title="Test Song",
            artist="Test Artist",
            tempo=120
        )
        
        assert song.title == "Test Song"
        assert song.artist == "Test Artist"
        assert song.tempo == 120
    
    def test_default_tuning(self):
        """Test default tuning"""
        song = TabSong()
        assert song.tuning == ['E', 'B', 'G', 'D', 'A', 'E']
    
    def test_custom_tuning(self):
        """Test setting custom tuning"""
        song = TabSong()
        song.set_tuning(['E', 'C', 'G', 'D', 'G', 'C'])
        
        assert song.tuning == ['E', 'C', 'G', 'D', 'G', 'C']
    
    def test_add_measures(self):
        """Test adding measures"""
        song = TabSong()
        
        measure1 = TabMeasure(chords=[
            TabChord(notes=[TabNote(string=1, fret=0)])
        ])
        measure2 = TabMeasure(chords=[
            TabChord(notes=[TabNote(string=2, fret=3)])
        ])
        
        song.add_measure(measure1)
        song.add_measure(measure2)
        
        assert song.measure_count == 2
        assert song.measures[0].number == 1
        assert song.measures[1].number == 2
    
    def test_get_stats(self):
        """Test getting song stats"""
        song = TabSong(title="Test", tempo=100)
        
        measure = TabMeasure()
        measure.add_chord(TabChord(notes=[
            TabNote(string=1, fret=5, technique=Technique.HAMMER_ON),
            TabNote(string=2, fret=5)
        ]))
        measure.add_chord(TabChord(notes=[
            TabNote(string=1, fret=7)
        ]))
        
        song.add_measure(measure)
        
        stats = song.get_stats()
        assert stats['title'] == "Test"
        assert stats['tempo'] == 100
        assert stats['measures'] == 1
        assert stats['chords'] == 2
        assert stats['notes'] == 3
        assert stats['techniques']['h'] == 1


class TestChordGrouper:
    """Test ChordGrouper"""
    
    def test_group_single_notes(self):
        """Test grouping single notes"""
        from omnitab.tab_ocr.parser import ChordGrouper
        
        notes = [
            TabNote(string=1, fret=5, x_position=10),
            TabNote(string=1, fret=7, x_position=50),
            TabNote(string=1, fret=5, x_position=90),
        ]
        
        grouper = ChordGrouper(x_threshold=15)
        chords = grouper.group_notes(notes)
        
        assert len(chords) == 3
        assert all(chord.is_single_note for chord in chords)
    
    def test_group_chord(self):
        """Test grouping notes into chord"""
        from omnitab.tab_ocr.parser import ChordGrouper
        
        notes = [
            TabNote(string=1, fret=0, x_position=10),
            TabNote(string=2, fret=1, x_position=12),
            TabNote(string=3, fret=0, x_position=11),
        ]
        
        grouper = ChordGrouper(x_threshold=15)
        chords = grouper.group_notes(notes)
        
        assert len(chords) == 1
        assert chords[0].is_chord
        assert len(chords[0].notes) == 3
    
    def test_validate_chord(self):
        """Test chord validation"""
        from omnitab.tab_ocr.parser import ChordGrouper
        
        grouper = ChordGrouper()
        
        # Valid chord
        valid_chord = TabChord(notes=[
            TabNote(string=1, fret=5),
            TabNote(string=2, fret=5),
            TabNote(string=3, fret=6),
        ])
        assert grouper.validate_chord(valid_chord)
        
        # Invalid - duplicate string
        invalid_chord = TabChord(notes=[
            TabNote(string=1, fret=5),
            TabNote(string=1, fret=7),  # Same string!
        ])
        assert not grouper.validate_chord(invalid_chord)


class TestTimingAnalyzer:
    """Test TimingAnalyzer"""
    
    def test_uniform_timing(self):
        """Test uniform timing analysis"""
        from omnitab.tab_ocr.parser import TimingAnalyzer
        
        chords = [
            TabChord(notes=[TabNote(string=1, fret=5)], x_position=10),
            TabChord(notes=[TabNote(string=1, fret=7)], x_position=30),
            TabChord(notes=[TabNote(string=1, fret=5)], x_position=50),
        ]
        
        analyzer = TimingAnalyzer()
        result = analyzer.analyze_uniform(chords, Duration.EIGHTH)
        
        assert all(chord.duration == Duration.EIGHTH for chord in result)
    
    def test_proportional_timing(self):
        """Test proportional timing analysis"""
        from omnitab.tab_ocr.parser import TimingAnalyzer
        
        # Varying gaps
        chords = [
            TabChord(notes=[TabNote(string=1, fret=5)], x_position=0),
            TabChord(notes=[TabNote(string=1, fret=7)], x_position=10),   # Small gap
            TabChord(notes=[TabNote(string=1, fret=5)], x_position=50),   # Large gap
        ]
        
        analyzer = TimingAnalyzer()
        result = analyzer.analyze_proportional(chords)
        
        # First note should have shorter duration (smaller gap after it)
        # Second note should have longer duration (larger gap after it)
        assert result[0].duration.value > result[1].duration.value


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
