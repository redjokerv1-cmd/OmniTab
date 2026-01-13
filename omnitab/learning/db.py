"""
Learning Database - SQLite-based storage for all OCR attempts and patterns
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from .models import OCRAttempt, ErrorPattern, Correction


class LearningDB:
    """
    SQLite database for storing learning data.
    
    Tables:
    - ocr_attempts: All conversion attempts
    - error_patterns: Identified error patterns
    - corrections: User corrections
    - settings_history: Settings that worked well
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "learning.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # OCR Attempts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_attempts (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                image_path TEXT,
                image_hash TEXT,
                total_digits INTEGER,
                mapped_digits INTEGER,
                unmapped_digits INTEGER,
                duplicates_removed INTEGER,
                systems_detected INTEGER,
                measures_detected INTEGER,
                avg_confidence REAL,
                suspicious_count INTEGER,
                gp5_path TEXT,
                gp5_notes INTEGER,
                gp5_measures INTEGER,
                settings TEXT,
                user_rating INTEGER,
                user_notes TEXT
            )
        ''')
        
        # Error Patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT,
                description TEXT,
                image_region TEXT,
                expected_value TEXT,
                actual_value TEXT,
                x REAL,
                y REAL,
                string INTEGER,
                occurrences INTEGER,
                resolved INTEGER,
                resolution TEXT
            )
        ''')
        
        # Corrections
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                attempt_id TEXT,
                timestamp TEXT,
                original_string INTEGER,
                original_fret INTEGER,
                original_x REAL,
                original_y REAL,
                correct_string INTEGER,
                correct_fret INTEGER,
                measure_num INTEGER,
                beat_num INTEGER,
                FOREIGN KEY (attempt_id) REFERENCES ocr_attempts(id)
            )
        ''')
        
        # Insights (learned patterns)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                insight_type TEXT,
                description TEXT,
                data TEXT,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # === OCR Attempts ===
    
    def save_attempt(self, attempt: OCRAttempt) -> str:
        """Save an OCR attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO ocr_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            attempt.id,
            attempt.timestamp.isoformat(),
            attempt.image_path,
            attempt.image_hash,
            attempt.total_digits,
            attempt.mapped_digits,
            attempt.unmapped_digits,
            attempt.duplicates_removed,
            attempt.systems_detected,
            attempt.measures_detected,
            attempt.avg_confidence,
            attempt.suspicious_count,
            attempt.gp5_path,
            attempt.gp5_notes,
            attempt.gp5_measures,
            json.dumps(attempt.settings),
            attempt.user_rating,
            attempt.user_notes
        ))
        
        conn.commit()
        conn.close()
        
        return attempt.id
    
    def get_attempts_for_image(self, image_hash: str) -> List[OCRAttempt]:
        """Get all attempts for a specific image"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM ocr_attempts WHERE image_hash = ?', (image_hash,))
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_attempt(row) for row in rows]
    
    def get_best_settings_for_image(self, image_hash: str) -> Optional[Dict]:
        """Get settings that produced best results for similar images"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get attempt with highest user rating or mapping ratio
        cursor.execute('''
            SELECT settings FROM ocr_attempts 
            WHERE image_hash = ? AND user_rating IS NOT NULL
            ORDER BY user_rating DESC, mapped_digits * 1.0 / total_digits DESC
            LIMIT 1
        ''', (image_hash,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None
    
    def get_all_attempts(self, limit: int = 100) -> List[OCRAttempt]:
        """Get recent attempts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM ocr_attempts ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_attempt(row) for row in rows]
    
    # === Error Patterns ===
    
    def save_error_pattern(self, pattern: ErrorPattern) -> str:
        """Save an error pattern"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO error_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.id,
            pattern.pattern_type,
            pattern.description,
            pattern.image_region,
            pattern.expected_value,
            pattern.actual_value,
            pattern.x,
            pattern.y,
            pattern.string,
            pattern.occurrences,
            1 if pattern.resolved else 0,
            pattern.resolution
        ))
        
        conn.commit()
        conn.close()
        
        return pattern.id
    
    def get_unresolved_patterns(self) -> List[ErrorPattern]:
        """Get error patterns that haven't been resolved"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM error_patterns WHERE resolved = 0 ORDER BY occurrences DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_pattern(row) for row in rows]
    
    # === Corrections ===
    
    def save_correction(self, correction: Correction) -> str:
        """Save a user correction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO corrections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            correction.id,
            correction.attempt_id,
            correction.timestamp.isoformat(),
            correction.original_string,
            correction.original_fret,
            correction.original_x,
            correction.original_y,
            correction.correct_string,
            correction.correct_fret,
            correction.measure_num,
            correction.beat_num
        ))
        
        conn.commit()
        conn.close()
        
        return correction.id
    
    # === Insights ===
    
    def save_insight(self, insight_type: str, description: str, data: Dict, confidence: float = 0.5):
        """Save a learned insight"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            datetime.now().isoformat(),
            insight_type,
            description,
            json.dumps(data),
            confidence
        ))
        
        conn.commit()
        conn.close()
    
    def get_insights(self, insight_type: str = None) -> List[Dict]:
        """Get learned insights"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if insight_type:
            cursor.execute('SELECT * FROM insights WHERE insight_type = ? ORDER BY confidence DESC', (insight_type,))
        else:
            cursor.execute('SELECT * FROM insights ORDER BY confidence DESC')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'id': r[0],
            'created_at': r[1],
            'type': r[2],
            'description': r[3],
            'data': json.loads(r[4]),
            'confidence': r[5]
        } for r in rows]
    
    # === Stats ===
    
    def get_stats(self) -> Dict:
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM ocr_attempts')
        total_attempts = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(mapped_digits * 1.0 / total_digits) FROM ocr_attempts WHERE total_digits > 0')
        avg_mapping_rate = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT COUNT(*) FROM error_patterns WHERE resolved = 0')
        unresolved_patterns = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM corrections')
        total_corrections = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM insights')
        total_insights = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_attempts': total_attempts,
            'avg_mapping_rate': avg_mapping_rate,
            'unresolved_patterns': unresolved_patterns,
            'total_corrections': total_corrections,
            'total_insights': total_insights
        }
    
    # === Helpers ===
    
    def _row_to_attempt(self, row) -> OCRAttempt:
        return OCRAttempt(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            image_path=row[2],
            image_hash=row[3],
            total_digits=row[4],
            mapped_digits=row[5],
            unmapped_digits=row[6],
            duplicates_removed=row[7],
            systems_detected=row[8],
            measures_detected=row[9],
            avg_confidence=row[10],
            suspicious_count=row[11],
            gp5_path=row[12],
            gp5_notes=row[13],
            gp5_measures=row[14],
            settings=json.loads(row[15]) if row[15] else {},
            user_rating=row[16],
            user_notes=row[17]
        )
    
    def _row_to_pattern(self, row) -> ErrorPattern:
        return ErrorPattern(
            id=row[0],
            pattern_type=row[1],
            description=row[2],
            image_region=row[3],
            expected_value=row[4],
            actual_value=row[5],
            x=row[6],
            y=row[7],
            string=row[8],
            occurrences=row[9],
            resolved=row[10] == 1,
            resolution=row[11]
        )


# Utility
def get_image_hash(image_path: str) -> str:
    """Get hash of image file for identification"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
