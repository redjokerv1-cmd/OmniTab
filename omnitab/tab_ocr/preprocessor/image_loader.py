"""ImageLoader - Load PDF and image files"""

import os
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


@dataclass
class PageImage:
    """Represents a single page image"""
    image: np.ndarray
    page_number: int
    source_file: str
    width: int
    height: int


class ImageLoader:
    """
    Load images from PDF files or image files.
    
    Supports:
    - PDF files (via PyMuPDF)
    - Image files (PNG, JPG, etc.)
    """
    
    def __init__(self, dpi: int = 300):
        """
        Initialize ImageLoader.
        
        Args:
            dpi: Resolution for PDF rendering (default 300)
        """
        self.dpi = dpi
        
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")
    
    def load(self, file_path: Union[str, Path]) -> List[PageImage]:
        """
        Load images from file.
        
        Args:
            file_path: Path to PDF or image file
            
        Returns:
            List of PageImage objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._load_pdf(file_path)
        elif extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            return self._load_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _load_pdf(self, pdf_path: Path) -> List[PageImage]:
        """Load all pages from PDF"""
        if fitz is None:
            raise ImportError("PyMuPDF is required for PDF. Install with: pip install PyMuPDF")
        
        pages = []
        doc = fitz.open(str(pdf_path))
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render at specified DPI
            zoom = self.dpi / 72  # 72 is default PDF DPI
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to numpy array
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_data.reshape(pix.height, pix.width, pix.n)
            
            # Convert to BGR for OpenCV compatibility
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            pages.append(PageImage(
                image=img,
                page_number=page_num + 1,
                source_file=str(pdf_path),
                width=img.shape[1],
                height=img.shape[0]
            ))
        
        doc.close()
        return pages
    
    def _load_image(self, image_path: Path) -> List[PageImage]:
        """Load single image file"""
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return [PageImage(
            image=img,
            page_number=1,
            source_file=str(image_path),
            width=img.shape[1],
            height=img.shape[0]
        )]
    
    def load_from_directory(self, dir_path: Union[str, Path], 
                           pattern: str = "*.png") -> List[PageImage]:
        """
        Load all images from a directory.
        
        Args:
            dir_path: Directory path
            pattern: Glob pattern for files
            
        Returns:
            List of PageImage objects, sorted by filename
        """
        dir_path = Path(dir_path)
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        files = sorted(dir_path.glob(pattern))
        
        pages = []
        for i, file in enumerate(files):
            page_images = self._load_image(file)
            for page in page_images:
                page.page_number = i + 1
            pages.extend(page_images)
        
        return pages
