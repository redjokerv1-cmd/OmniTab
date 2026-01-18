"""
Synthetic TAB Data Generator for YOLO Training

Generates:
1. TAB images with random notes
2. YOLO format annotation files (class x_center y_center width height)

Classes:
- 0-24: fret numbers
- 25: 'h' (hammer-on)
- 26: 'p' (pull-off)
- 27: 'x' (muted)
- 28: '<>' (harmonic)
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import json


@dataclass
class TabNote:
    """A note on the TAB with its position"""
    string: int      # 1-6
    fret: int        # 0-24, or -1 for muted
    x: int           # pixel x position
    y: int           # pixel y position
    width: int       # text width
    height: int      # text height
    text: str        # display text (e.g., "12", "x", "<12>")
    class_id: int    # YOLO class ID


class SyntheticTabGenerator:
    """
    Generate synthetic TAB images with perfect annotations.
    
    Uses PIL to draw TAB images and records exact positions
    of all elements for YOLO training.
    """
    
    # Image settings
    DEFAULT_WIDTH = 800
    DEFAULT_HEIGHT = 150
    LINE_SPACING = 18
    TOP_MARGIN = 25
    LEFT_MARGIN = 40
    RIGHT_MARGIN = 20
    
    # Font settings
    FONT_SIZE = 14
    
    # Class mapping
    CLASS_MAP = {
        **{str(i): i for i in range(25)},  # 0-24: fret numbers
        'h': 25,
        'p': 26,
        'x': 27,
        'X': 27,
        'harmonic': 28,
    }
    
    def __init__(self, output_dir: str, font_path: Optional[str] = None):
        """
        Args:
            output_dir: Directory to save images and annotations
            font_path: Path to TTF font file (optional)
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Load font
        try:
            if font_path:
                self.font = ImageFont.truetype(font_path, self.FONT_SIZE)
            else:
                # Try common fonts
                for fname in ["arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"]:
                    try:
                        self.font = ImageFont.truetype(fname, self.FONT_SIZE)
                        break
                    except:
                        continue
                else:
                    self.font = ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()
    
    def _get_line_y(self, string_num: int) -> int:
        """Get Y coordinate for a string (1-6)"""
        return self.TOP_MARGIN + (string_num - 1) * self.LINE_SPACING
    
    def _get_text_bbox(self, draw: ImageDraw.Draw, text: str) -> Tuple[int, int]:
        """Get text width and height"""
        bbox = draw.textbbox((0, 0), text, font=self.font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    def _draw_tab_lines(self, draw: ImageDraw.Draw, width: int):
        """Draw 6 horizontal TAB lines"""
        for string in range(1, 7):
            y = self._get_line_y(string)
            draw.line([(self.LEFT_MARGIN, y), (width - self.RIGHT_MARGIN, y)], 
                     fill=(150, 150, 150), width=1)
    
    def _draw_tab_header(self, draw: ImageDraw.Draw):
        """Draw TAB header on left side"""
        header = ["T", "A", "B"]
        for i, char in enumerate(header):
            y = self._get_line_y(i + 2) - 8
            draw.text((5, y), char, fill=(100, 100, 100), font=self.font)
    
    def generate_random_measure(self, num_beats: int = 8) -> List[List[TabNote]]:
        """
        Generate random notes for one measure.
        
        Returns:
            List of beats, each beat is a list of TabNote objects
        """
        beats = []
        
        for beat_idx in range(num_beats):
            beat_notes = []
            
            # Random number of notes per beat (1-4)
            num_notes = random.choices([1, 2, 3, 4], weights=[0.3, 0.35, 0.25, 0.1])[0]
            
            # Random strings (no duplicates)
            strings = random.sample(range(1, 7), min(num_notes, 6))
            
            for string in strings:
                # Random fret (weighted towards lower frets)
                if random.random() < 0.85:
                    fret = random.choices(
                        range(0, 15),
                        weights=[0.15, 0.1, 0.1, 0.1, 0.08, 0.08, 0.07, 0.06, 
                                0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02]
                    )[0]
                else:
                    fret = random.randint(15, 24)
                
                # Random technique (10% chance)
                technique = None
                if random.random() < 0.05:
                    technique = random.choice(['h', 'p'])
                elif random.random() < 0.02:
                    technique = 'x'  # muted
                elif random.random() < 0.03:
                    technique = 'harmonic'
                
                # Create text and class
                if technique == 'x':
                    text = 'x'
                    class_id = self.CLASS_MAP['x']
                elif technique == 'harmonic':
                    text = f'<{fret}>'
                    class_id = self.CLASS_MAP['harmonic']
                else:
                    text = str(fret)
                    class_id = fret
                
                beat_notes.append(TabNote(
                    string=string,
                    fret=fret,
                    x=0,  # Will be set during rendering
                    y=0,
                    width=0,
                    height=0,
                    text=text,
                    class_id=class_id
                ))
            
            beats.append(beat_notes)
        
        return beats
    
    def render_tab_image(self, 
                        beats: List[List[TabNote]], 
                        width: Optional[int] = None) -> Tuple[Image.Image, List[TabNote]]:
        """
        Render TAB image and return all notes with exact positions.
        
        Returns:
            (PIL Image, list of TabNote with x,y,w,h filled in)
        """
        width = width or self.DEFAULT_WIDTH
        height = self.DEFAULT_HEIGHT
        
        # Create image
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw TAB lines
        self._draw_tab_lines(draw, width)
        
        # Draw header
        self._draw_tab_header(draw)
        
        # Calculate beat spacing
        usable_width = width - self.LEFT_MARGIN - self.RIGHT_MARGIN
        beat_spacing = usable_width / (len(beats) + 1)
        
        # Draw notes and collect annotations
        all_notes = []
        
        for beat_idx, beat_notes in enumerate(beats):
            x_center = self.LEFT_MARGIN + (beat_idx + 1) * beat_spacing
            
            for note in beat_notes:
                # Get text size
                text_w, text_h = self._get_text_bbox(draw, note.text)
                
                # Calculate position (centered on line)
                x = int(x_center - text_w / 2)
                y = self._get_line_y(note.string) - text_h // 2 - 2
                
                # Draw white background to cover line
                padding = 2
                draw.rectangle(
                    [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
                    fill=(255, 255, 255)
                )
                
                # Draw text
                draw.text((x, y), note.text, fill=(0, 0, 0), font=self.font)
                
                # Update note with position
                note.x = x
                note.y = y
                note.width = text_w
                note.height = text_h
                
                all_notes.append(note)
        
        # Draw measure line at end
        x_end = width - self.RIGHT_MARGIN
        y_top = self._get_line_y(1)
        y_bottom = self._get_line_y(6)
        draw.line([(x_end, y_top), (x_end, y_bottom)], fill=(0, 0, 0), width=1)
        
        return img, all_notes
    
    def notes_to_yolo(self, notes: List[TabNote], 
                      img_width: int, img_height: int) -> str:
        """
        Convert notes to YOLO format annotation.
        
        YOLO format: class_id x_center y_center width height
        All values normalized to 0-1.
        """
        lines = []
        
        for note in notes:
            # Calculate center
            x_center = (note.x + note.width / 2) / img_width
            y_center = (note.y + note.height / 2) / img_height
            w = note.width / img_width
            h = note.height / img_height
            
            # Clamp to valid range
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0.001, min(1, w))
            h = max(0.001, min(1, h))
            
            line = f"{note.class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def add_augmentation(self, img: Image.Image, strength: float = 0.3) -> Image.Image:
        """
        Add realistic augmentations to make synthetic data closer to real scans.
        
        Args:
            img: PIL Image
            strength: Augmentation strength (0-1)
        """
        from PIL import ImageFilter, ImageEnhance
        import random
        
        # Random brightness
        if random.random() < strength:
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.85, 1.15)
            img = enhancer.enhance(factor)
        
        # Random contrast
        if random.random() < strength:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.9, 1.1)
            img = enhancer.enhance(factor)
        
        # Slight blur (simulating scan quality)
        if random.random() < strength * 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        
        # Add noise
        if random.random() < strength * 0.3:
            import numpy as np
            arr = np.array(img)
            noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        
        return img
    
    def generate_dataset(self, 
                        num_samples: int = 1000,
                        beats_range: Tuple[int, int] = (4, 12),
                        width_range: Tuple[int, int] = (600, 1000),
                        augment: bool = True,
                        verbose: bool = True) -> Dict:
        """
        Generate a full dataset of synthetic TAB images.
        
        Args:
            num_samples: Number of images to generate
            beats_range: (min, max) beats per image
            width_range: (min, max) image width
            verbose: Print progress
            
        Returns:
            Statistics dict
        """
        stats = {
            'total_images': 0,
            'total_notes': 0,
            'class_counts': {i: 0 for i in range(29)}
        }
        
        for i in range(num_samples):
            # Random parameters
            num_beats = random.randint(*beats_range)
            width = random.randint(*width_range)
            
            # Generate random measure
            beats = self.generate_random_measure(num_beats)
            
            # Render image
            img, notes = self.render_tab_image(beats, width)
            
            # Apply augmentation
            if augment and random.random() < 0.7:
                img = self.add_augmentation(img, strength=0.3)
            
            # Generate annotation
            annotation = self.notes_to_yolo(notes, img.width, img.height)
            
            # Save files
            filename = f"tab_{i:06d}"
            img.save(self.images_dir / f"{filename}.png")
            
            with open(self.labels_dir / f"{filename}.txt", 'w') as f:
                f.write(annotation)
            
            # Update stats
            stats['total_images'] += 1
            stats['total_notes'] += len(notes)
            for note in notes:
                stats['class_counts'][note.class_id] += 1
            
            if verbose and (i + 1) % 100 == 0:
                print(f"[Generator] Progress: {i + 1}/{num_samples}")
        
        # Save class names
        class_names = [str(i) for i in range(25)] + ['h', 'p', 'x', 'harmonic']
        with open(self.output_dir / "classes.txt", 'w') as f:
            f.write("\n".join(class_names))
        
        # Save YOLO data.yaml
        yaml_content = f"""
path: {self.output_dir.absolute()}
train: images
val: images
nc: 29
names: {class_names}
"""
        with open(self.output_dir / "data.yaml", 'w') as f:
            f.write(yaml_content.strip())
        
        # Save stats
        with open(self.output_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        if verbose:
            print(f"\n[Generator] Complete!")
            print(f"  Images: {stats['total_images']}")
            print(f"  Total notes: {stats['total_notes']}")
            print(f"  Saved to: {self.output_dir}")
        
        return stats


def main():
    """Generate training dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic TAB training data")
    parser.add_argument("--output", "-o", default="training_data", 
                       help="Output directory")
    parser.add_argument("--samples", "-n", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--min-beats", type=int, default=4)
    parser.add_argument("--max-beats", type=int, default=12)
    
    args = parser.parse_args()
    
    generator = SyntheticTabGenerator(args.output)
    stats = generator.generate_dataset(
        num_samples=args.samples,
        beats_range=(args.min_beats, args.max_beats)
    )
    
    print(f"\nGenerated {stats['total_images']} images with {stats['total_notes']} notes")


if __name__ == "__main__":
    main()
