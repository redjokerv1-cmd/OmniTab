"""
Realistic TAB Data Generator for YOLO Training

개선된 버전: 실제 TAB 스타일에 맞게 다양성 추가
- 다양한 폰트 및 크기
- 배경 노이즈 및 스캔 효과
- 다양한 레이아웃
- 오선보 + TAB 동시 표기
- 실제 악보처럼 보이는 스타일
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import json


@dataclass
class TabNote:
    """A note on the TAB with its position"""
    string: int      # 1-6
    fret: int        # 0-24
    x: int           # pixel x position
    y: int           # pixel y position
    width: int       # text width
    height: int      # text height
    text: str        # display text
    class_id: int    # YOLO class ID


class RealisticTabGenerator:
    """
    Generate realistic TAB images that match real-world TAB styles.
    """
    
    # Class mapping (same as YOLO training)
    CLASS_MAP = {
        **{str(i): i for i in range(25)},  # 0-24: fret numbers
        'h': 25,
        'p': 26,
        'x': 27,
        'X': 27,
        'harmonic': 28,
    }
    
    # Font variations
    FONT_NAMES = [
        "arial.ttf",
        "arialbd.ttf",      # Arial Bold
        "times.ttf",        # Times New Roman
        "timesbd.ttf",      # Times Bold
        "cour.ttf",         # Courier
        "courbd.ttf",       # Courier Bold
        "calibri.ttf",
        "cambria.ttc",
        "consola.ttf",      # Consolas
        "verdana.ttf",
        "tahoma.ttf",
    ]
    
    FONT_SIZES = [12, 13, 14, 15, 16, 18]
    
    # Layout variations
    LINE_SPACINGS = [16, 18, 20, 22, 24]
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Available fonts
        self.fonts = []
        for fname in self.FONT_NAMES:
            for size in self.FONT_SIZES:
                try:
                    font = ImageFont.truetype(fname, size)
                    self.fonts.append((font, size, fname))
                except:
                    pass
        
        if not self.fonts:
            default_font = ImageFont.load_default()
            self.fonts = [(default_font, 12, "default")]
        
        print(f"[RealisticGen] Loaded {len(self.fonts)} font variations")
    
    def _random_style(self) -> Dict:
        """랜덤 스타일 생성"""
        font, size, name = random.choice(self.fonts)
        
        return {
            'font': font,
            'font_size': size,
            'font_name': name,
            'line_spacing': random.choice(self.LINE_SPACINGS),
            'bg_color': self._random_bg_color(),
            'text_color': self._random_text_color(),
            'line_color': self._random_line_color(),
            'add_noise': random.random() < 0.5,
            'add_blur': random.random() < 0.2,
            'add_rotation': random.random() < 0.1,
            'add_staff': random.random() < 0.4,  # 40% chance of staff notation above
        }
    
    def _random_bg_color(self) -> Tuple[int, int, int]:
        """다양한 배경색"""
        choices = [
            (255, 255, 255),      # White
            (250, 250, 250),      # Light gray
            (255, 253, 245),      # Cream
            (245, 245, 240),      # Off-white
            (252, 252, 248),      # Paper
            (248, 246, 242),      # Aged paper
            (255, 255, 250),      # Ivory
        ]
        return random.choice(choices)
    
    def _random_text_color(self) -> Tuple[int, int, int]:
        """다양한 텍스트 색상"""
        choices = [
            (0, 0, 0),            # Black
            (20, 20, 20),         # Near black
            (40, 40, 40),         # Dark gray
            (50, 50, 60),         # Bluish black
        ]
        return random.choice(choices)
    
    def _random_line_color(self) -> Tuple[int, int, int]:
        """줄 색상"""
        choices = [
            (120, 120, 120),      # Gray
            (150, 150, 150),      # Light gray
            (100, 100, 100),      # Dark gray
            (80, 80, 80),         # Very dark
        ]
        return random.choice(choices)
    
    def _add_noise(self, img: Image.Image, intensity: float = 0.03) -> Image.Image:
        """노이즈 추가 (스캔 효과)"""
        np_img = np.array(img)
        noise = np.random.normal(0, intensity * 255, np_img.shape).astype(np.int16)
        noisy = np.clip(np_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def _add_subtle_rotation(self, img: Image.Image, max_angle: float = 0.5) -> Image.Image:
        """미세한 회전 (스캔 효과)"""
        angle = random.uniform(-max_angle, max_angle)
        return img.rotate(angle, fillcolor=img.getpixel((0, 0)), expand=False)
    
    def _draw_staff_notation(self, draw: ImageDraw.Draw, y_offset: int, 
                             width: int, num_beats: int, style: Dict):
        """오선보 그리기 (TAB 위에)"""
        staff_spacing = 8
        staff_top = y_offset
        
        # 5 lines for staff
        for i in range(5):
            y = staff_top + i * staff_spacing
            draw.line([(40, y), (width - 20, y)], fill=style['line_color'], width=1)
        
        # Add some random note heads (just for visual variety)
        beat_spacing = (width - 60) / (num_beats + 1)
        for beat_idx in range(num_beats):
            x = 40 + (beat_idx + 1) * beat_spacing
            # Random position on staff
            y = staff_top + random.randint(0, 4) * staff_spacing
            
            # Draw note head (ellipse)
            draw.ellipse([x-3, y-2, x+3, y+2], fill=style['text_color'])
            
            # Draw stem (sometimes)
            if random.random() < 0.8:
                stem_dir = random.choice([-1, 1])
                draw.line([(x+3 if stem_dir > 0 else x-3, y), 
                          (x+3 if stem_dir > 0 else x-3, y - 25*stem_dir)],
                         fill=style['text_color'], width=1)
    
    def _draw_tab_lines(self, draw: ImageDraw.Draw, y_offset: int, 
                        width: int, style: Dict) -> List[int]:
        """TAB 줄 그리기"""
        line_ys = []
        for i in range(6):
            y = y_offset + i * style['line_spacing']
            line_ys.append(y)
            draw.line([(40, y), (width - 20, y)], fill=style['line_color'], width=1)
        
        # TAB header
        header_y = line_ys[1]
        draw.text((10, header_y - 8), "T", fill=style['text_color'], font=style['font'])
        draw.text((10, header_y + style['line_spacing'] - 8), "A", 
                 fill=style['text_color'], font=style['font'])
        draw.text((10, header_y + 2*style['line_spacing'] - 8), "B", 
                 fill=style['text_color'], font=style['font'])
        
        return line_ys
    
    def generate_beats(self, num_beats: int = 8) -> List[List[Dict]]:
        """랜덤 비트 생성"""
        beats = []
        
        for _ in range(num_beats):
            beat_notes = []
            num_notes = random.choices([1, 2, 3, 4, 5], weights=[0.25, 0.30, 0.25, 0.15, 0.05])[0]
            
            strings = random.sample(range(1, 7), min(num_notes, 6))
            
            for string in strings:
                # 실제 TAB에서 자주 나오는 프렛 분포
                if random.random() < 0.2:
                    fret = 0  # Open string (20%)
                elif random.random() < 0.7:
                    fret = random.choices(
                        list(range(1, 13)),
                        weights=[0.08, 0.10, 0.10, 0.08, 0.10, 0.05, 0.10, 0.05, 0.10, 0.08, 0.08, 0.08]
                    )[0]
                else:
                    fret = random.randint(13, 24)
                
                # Technique
                technique = None
                if random.random() < 0.03:
                    technique = 'harmonic'
                elif random.random() < 0.02:
                    technique = 'x'
                
                # Text and class
                if technique == 'x':
                    text = 'x'
                    class_id = self.CLASS_MAP['x']
                elif technique == 'harmonic':
                    text = f'<{fret}>'
                    class_id = self.CLASS_MAP['harmonic']
                else:
                    text = str(fret)
                    class_id = fret
                
                beat_notes.append({
                    'string': string,
                    'fret': fret,
                    'text': text,
                    'class_id': class_id,
                    'technique': technique
                })
            
            beats.append(beat_notes)
        
        return beats
    
    def render_image(self, beats: List[List[Dict]], style: Dict = None) -> Tuple[Image.Image, List[TabNote]]:
        """이미지 렌더링"""
        style = style or self._random_style()
        
        num_beats = len(beats)
        
        # Size calculation
        width = max(600, 50 + num_beats * 50)
        
        # Height depends on whether we have staff
        staff_height = 50 if style['add_staff'] else 0
        tab_height = 6 * style['line_spacing'] + 40
        height = staff_height + tab_height
        
        # Create image
        img = Image.new('RGB', (width, height), color=style['bg_color'])
        draw = ImageDraw.Draw(img)
        
        # Draw staff if needed
        if style['add_staff']:
            self._draw_staff_notation(draw, 10, width, num_beats, style)
        
        # Draw TAB
        tab_y_offset = staff_height + 20
        line_ys = self._draw_tab_lines(draw, tab_y_offset, width, style)
        
        # Draw notes
        beat_spacing = (width - 60) / (num_beats + 1)
        all_notes = []
        
        for beat_idx, beat_notes in enumerate(beats):
            x_center = 40 + (beat_idx + 1) * beat_spacing
            
            for note_data in beat_notes:
                string = note_data['string']
                text = note_data['text']
                class_id = note_data['class_id']
                
                # Get text size
                bbox = draw.textbbox((0, 0), text, font=style['font'])
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # Position
                x = int(x_center - text_w / 2)
                y = line_ys[string - 1] - text_h // 2 - 2
                
                # White background to cover line
                padding = 2
                draw.rectangle(
                    [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
                    fill=style['bg_color']
                )
                
                # Draw text
                draw.text((x, y), text, fill=style['text_color'], font=style['font'])
                
                all_notes.append(TabNote(
                    string=string,
                    fret=note_data['fret'],
                    x=x,
                    y=y,
                    width=text_w + 2 * padding,
                    height=text_h + 2 * padding,
                    text=text,
                    class_id=class_id
                ))
        
        # Draw measure lines
        for i in range(1, (num_beats // 4) + 1):
            x = 40 + i * 4 * beat_spacing
            draw.line([(x, line_ys[0]), (x, line_ys[-1])], fill=style['line_color'], width=1)
        
        # End bar
        x_end = width - 20
        draw.line([(x_end, line_ys[0]), (x_end, line_ys[-1])], fill=(0, 0, 0), width=2)
        
        # Post-processing
        if style['add_noise']:
            img = self._add_noise(img, intensity=0.02)
        
        if style['add_blur']:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        if style['add_rotation']:
            img = self._add_subtle_rotation(img)
        
        return img, all_notes
    
    def generate_one(self, idx: int, num_beats: int = 8) -> Tuple[str, str]:
        """단일 샘플 생성"""
        beats = self.generate_beats(num_beats)
        style = self._random_style()
        img, notes = self.render_image(beats, style)
        
        # Save image
        img_path = self.images_dir / f"tab_{idx:06d}.png"
        img.save(img_path)
        
        # Save YOLO annotation
        txt_path = self.labels_dir / f"tab_{idx:06d}.txt"
        with open(txt_path, 'w') as f:
            for note in notes:
                # YOLO format: class x_center y_center width height (normalized)
                x_center = (note.x + note.width / 2) / img.width
                y_center = (note.y + note.height / 2) / img.height
                w = note.width / img.width
                h = note.height / img.height
                f.write(f"{note.class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        return str(img_path), str(txt_path)
    
    def generate_dataset(self, num_samples: int = 1000, 
                        min_beats: int = 4, max_beats: int = 16) -> None:
        """전체 데이터셋 생성"""
        print(f"[RealisticGen] Generating {num_samples} samples...")
        
        for i in range(num_samples):
            num_beats = random.randint(min_beats, max_beats)
            self.generate_one(i, num_beats)
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_samples} generated")
        
        # Create data.yaml
        self._create_data_yaml()
        
        print(f"[RealisticGen] Done! {num_samples} samples in {self.output_dir}")
    
    def _create_data_yaml(self):
        """YOLO data.yaml 생성"""
        yaml_content = f"""path: {self.output_dir.resolve()}
train: images
val: images

nc: 29
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', 'h', 'p', 'x', 'harmonic']
"""
        (self.output_dir / "data.yaml").write_text(yaml_content)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate realistic TAB training data")
    parser.add_argument("--output", "-o", default="training_data_realistic",
                       help="Output directory")
    parser.add_argument("--samples", "-n", type=int, default=2000,
                       help="Number of samples to generate")
    
    args = parser.parse_args()
    
    generator = RealisticTabGenerator(args.output)
    generator.generate_dataset(args.samples)
