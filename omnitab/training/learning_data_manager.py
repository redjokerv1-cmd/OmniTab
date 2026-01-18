"""
Learning Data Manager - Accumulate and version training data

Key features:
1. Store real TAB images with manual/corrected annotations
2. Track data versions and model performance
3. Enable active learning loop (predict → correct → retrain)
4. Separate synthetic data from real data
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3


@dataclass
class AnnotatedImage:
    """A real TAB image with its annotation"""
    image_hash: str          # SHA256 of image file
    image_path: str          # Relative path to image
    annotation_path: str     # Relative path to YOLO annotation
    source: str              # "manual", "corrected", "synthetic"
    created_at: str          # ISO timestamp
    notes: str = ""          # Optional notes
    verified: bool = False   # Human verified?
    model_version: str = ""  # Which model was used for initial prediction


class LearningDataManager:
    """
    Manages the accumulation of training data over time.
    
    Structure:
        learning_data/
        ├── real/                    # Real TAB images (IMPORTANT - backup!)
        │   ├── images/
        │   └── labels/
        ├── corrected/               # User-corrected predictions
        │   ├── images/
        │   └── labels/
        ├── synthetic/               # Generated data (can regenerate)
        │   ├── images/
        │   └── labels/
        ├── metadata.db              # SQLite database for tracking
        └── versions/                # Model checkpoints
    """
    
    def __init__(self, base_dir: str = "learning_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["real/images", "real/labels", 
                       "corrected/images", "corrected/labels",
                       "synthetic/images", "synthetic/labels",
                       "versions"]:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.base_dir / "metadata.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                image_path TEXT,
                annotation_path TEXT,
                source TEXT,
                created_at TEXT,
                notes TEXT,
                verified INTEGER DEFAULT 0,
                model_version TEXT
            )
        """)
        
        # Training runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                started_at TEXT,
                completed_at TEXT,
                num_images INTEGER,
                num_synthetic INTEGER,
                num_real INTEGER,
                num_corrected INTEGER,
                epochs INTEGER,
                final_map50 REAL,
                model_path TEXT,
                notes TEXT
            )
        """)
        
        # Corrections table (track what was corrected)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT,
                original_annotation TEXT,
                corrected_annotation TEXT,
                corrected_at TEXT,
                correction_type TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # First 16 chars
    
    def add_real_image(self, image_path: str, annotation_path: str,
                       notes: str = "", verified: bool = False) -> str:
        """
        Add a real TAB image with its annotation to the learning data.
        
        This is PRECIOUS DATA that cannot be regenerated!
        
        Args:
            image_path: Path to the TAB image
            annotation_path: Path to YOLO format annotation
            notes: Optional notes about this image
            verified: Whether annotation has been human-verified
            
        Returns:
            Image hash (unique identifier)
        """
        image_hash = self._compute_hash(image_path)
        timestamp = datetime.now().isoformat()
        
        # Copy files to learning_data/real/
        new_image_path = f"real/images/{image_hash}.png"
        new_annotation_path = f"real/labels/{image_hash}.txt"
        
        shutil.copy2(image_path, self.base_dir / new_image_path)
        shutil.copy2(annotation_path, self.base_dir / new_annotation_path)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO images (image_hash, image_path, annotation_path,
                                   source, created_at, notes, verified, model_version)
                VALUES (?, ?, ?, 'real', ?, ?, ?, '')
            """, (image_hash, new_image_path, new_annotation_path,
                  timestamp, notes, 1 if verified else 0))
            conn.commit()
        except sqlite3.IntegrityError:
            print(f"[LDM] Image already exists: {image_hash}")
        finally:
            conn.close()
        
        print(f"[LDM] Added real image: {image_hash}")
        return image_hash
    
    def add_corrected_prediction(self, image_path: str, 
                                 original_annotation: str,
                                 corrected_annotation: str,
                                 model_version: str = "") -> str:
        """
        Add a user-corrected prediction.
        
        This creates a feedback loop for model improvement!
        
        Args:
            image_path: Path to the image that was predicted
            original_annotation: What the model predicted (YOLO format string)
            corrected_annotation: What the user corrected it to
            model_version: Which model made the prediction
            
        Returns:
            Image hash
        """
        image_hash = self._compute_hash(image_path)
        timestamp = datetime.now().isoformat()
        
        # Copy image
        new_image_path = f"corrected/images/{image_hash}.png"
        new_annotation_path = f"corrected/labels/{image_hash}.txt"
        
        shutil.copy2(image_path, self.base_dir / new_image_path)
        
        # Save corrected annotation
        with open(self.base_dir / new_annotation_path, 'w') as f:
            f.write(corrected_annotation)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO images (image_hash, image_path, annotation_path,
                                   source, created_at, notes, verified, model_version)
                VALUES (?, ?, ?, 'corrected', ?, '', 1, ?)
            """, (image_hash, new_image_path, new_annotation_path,
                  timestamp, model_version))
            
            # Also save the correction history
            cursor.execute("""
                INSERT INTO corrections (image_hash, original_annotation,
                                        corrected_annotation, corrected_at, correction_type)
                VALUES (?, ?, ?, ?, 'manual')
            """, (image_hash, original_annotation, corrected_annotation, timestamp))
            
            conn.commit()
        except sqlite3.IntegrityError:
            # Update existing
            cursor.execute("""
                UPDATE images SET annotation_path = ?, verified = 1
                WHERE image_hash = ?
            """, (new_annotation_path, image_hash))
            conn.commit()
        finally:
            conn.close()
        
        print(f"[LDM] Added corrected prediction: {image_hash}")
        return image_hash
    
    def get_training_data_paths(self, include_synthetic: bool = True,
                                include_real: bool = True,
                                include_corrected: bool = True,
                                verified_only: bool = False) -> Tuple[List[str], List[str]]:
        """
        Get paths to all training data.
        
        Returns:
            (list of image paths, list of label paths)
        """
        image_paths = []
        label_paths = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sources = []
        if include_real:
            sources.append('real')
        if include_corrected:
            sources.append('corrected')
        
        if sources:
            query = f"""
                SELECT image_path, annotation_path FROM images
                WHERE source IN ({','.join(['?' for _ in sources])})
            """
            if verified_only:
                query += " AND verified = 1"
            
            cursor.execute(query, sources)
            
            for img_path, lbl_path in cursor.fetchall():
                image_paths.append(str(self.base_dir / img_path))
                label_paths.append(str(self.base_dir / lbl_path))
        
        conn.close()
        
        # Add synthetic data
        if include_synthetic:
            synth_images = self.base_dir / "synthetic/images"
            synth_labels = self.base_dir / "synthetic/labels"
            
            for img in synth_images.glob("*.png"):
                lbl = synth_labels / f"{img.stem}.txt"
                if lbl.exists():
                    image_paths.append(str(img))
                    label_paths.append(str(lbl))
        
        return image_paths, label_paths
    
    def get_stats(self) -> Dict:
        """Get statistics about the learning data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count by source
        cursor.execute("""
            SELECT source, COUNT(*), SUM(verified) FROM images GROUP BY source
        """)
        for source, count, verified in cursor.fetchall():
            stats[f"{source}_count"] = count
            stats[f"{source}_verified"] = verified or 0
        
        # Count synthetic
        synth_images = list((self.base_dir / "synthetic/images").glob("*.png"))
        stats["synthetic_count"] = len(synth_images)
        
        # Training runs
        cursor.execute("SELECT COUNT(*) FROM training_runs")
        stats["training_runs"] = cursor.fetchone()[0]
        
        # Corrections
        cursor.execute("SELECT COUNT(*) FROM corrections")
        stats["total_corrections"] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def record_training_run(self, run_id: str, num_images: int,
                           num_synthetic: int, num_real: int, num_corrected: int,
                           epochs: int, final_map50: float, model_path: str,
                           notes: str = ""):
        """Record a training run for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO training_runs (run_id, started_at, completed_at,
                                       num_images, num_synthetic, num_real,
                                       num_corrected, epochs, final_map50,
                                       model_path, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, datetime.now().isoformat(), datetime.now().isoformat(),
              num_images, num_synthetic, num_real, num_corrected,
              epochs, final_map50, model_path, notes))
        
        conn.commit()
        conn.close()
    
    def export_for_training(self, output_dir: str,
                           include_synthetic: bool = True,
                           include_real: bool = True,
                           include_corrected: bool = True) -> str:
        """
        Export all data to YOLO training format.
        
        Creates:
            output_dir/
            ├── images/
            ├── labels/
            └── data.yaml
        """
        output_path = Path(output_dir)
        (output_path / "images").mkdir(parents=True, exist_ok=True)
        (output_path / "labels").mkdir(parents=True, exist_ok=True)
        
        image_paths, label_paths = self.get_training_data_paths(
            include_synthetic, include_real, include_corrected
        )
        
        # Copy files
        for i, (img, lbl) in enumerate(zip(image_paths, label_paths)):
            shutil.copy2(img, output_path / "images" / f"img_{i:06d}.png")
            shutil.copy2(lbl, output_path / "labels" / f"img_{i:06d}.txt")
        
        # Create data.yaml
        class_names = [str(i) for i in range(25)] + ['h', 'p', 'x', 'harmonic']
        yaml_content = f"""
path: {output_path.absolute()}
train: images
val: images
nc: 29
names: {class_names}
"""
        with open(output_path / "data.yaml", 'w') as f:
            f.write(yaml_content.strip())
        
        print(f"[LDM] Exported {len(image_paths)} images to {output_dir}")
        return str(output_path / "data.yaml")


def main():
    """CLI for managing learning data"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Learning Data Manager")
    subparsers = parser.add_subparsers(dest="command")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show data statistics")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add real image")
    add_parser.add_argument("--image", "-i", required=True)
    add_parser.add_argument("--annotation", "-a", required=True)
    add_parser.add_argument("--notes", "-n", default="")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export for training")
    export_parser.add_argument("--output", "-o", required=True)
    export_parser.add_argument("--no-synthetic", action="store_true")
    
    args = parser.parse_args()
    
    ldm = LearningDataManager()
    
    if args.command == "stats":
        stats = ldm.get_stats()
        print("\n=== Learning Data Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.command == "add":
        ldm.add_real_image(args.image, args.annotation, args.notes)
    
    elif args.command == "export":
        ldm.export_for_training(args.output, 
                               include_synthetic=not args.no_synthetic)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
