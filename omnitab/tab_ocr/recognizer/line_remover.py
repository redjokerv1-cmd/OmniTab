"""
Staff Line Remover - Remove horizontal TAB lines for better OCR

The horizontal lines in TAB notation connect digits, causing OCR to 
recognize them as one big blob instead of separate numbers.

Strategy:
1. Detect horizontal lines using morphological operations
2. Remove them from the image
3. Result: isolated digits that OCR can recognize individually
"""

from typing import Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class StaffLineRemover:
    """Remove horizontal staff lines from TAB images"""
    
    def __init__(self, 
                 kernel_length: int = 40,
                 line_iterations: int = 2,
                 repair_kernel: int = 3):
        """
        Initialize StaffLineRemover.
        
        Args:
            kernel_length: Horizontal kernel length for line detection
                          (longer = detects longer lines only)
            line_iterations: Morphology iterations (more = thicker line detection)
            repair_kernel: Kernel for repairing broken digits after line removal
        """
        self.kernel_length = kernel_length
        self.line_iterations = line_iterations
        self.repair_kernel = repair_kernel
        
        if cv2 is None:
            raise ImportError("OpenCV required: pip install opencv-python")
    
    def remove_lines(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove horizontal lines from image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (image_without_lines, detected_lines_mask)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Binary threshold (invert: lines become white)
        _, binary = cv2.threshold(
            gray, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Step 2: Create horizontal kernel (long and thin)
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.kernel_length, 1)
        )
        
        # Step 3: Detect horizontal lines using morphology
        # MORPH_OPEN = erosion followed by dilation
        # This removes small objects (digits) and keeps long horizontal lines
        detected_lines = cv2.morphologyEx(
            binary, 
            cv2.MORPH_OPEN, 
            horizontal_kernel, 
            iterations=self.line_iterations
        )
        
        # Step 4: Remove lines from original
        # Where lines were detected, set to white (background)
        result = gray.copy()
        result[detected_lines == 255] = 255
        
        # Step 5: Optional repair - close small gaps in digits
        if self.repair_kernel > 0:
            repair_k = cv2.getStructuringElement(
                cv2.MORPH_RECT, 
                (self.repair_kernel, self.repair_kernel)
            )
            # Invert, dilate slightly to close gaps, then invert back
            result_inv = cv2.bitwise_not(result)
            result_inv = cv2.dilate(result_inv, repair_k, iterations=1)
            result = cv2.bitwise_not(result_inv)
        
        return result, detected_lines
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Convenience method - just return the cleaned image"""
        result, _ = self.remove_lines(image)
        return result


class EnhancedOCRPreprocessor:
    """
    Complete preprocessing pipeline for TAB OCR:
    1. Line removal
    2. Contrast enhancement
    3. Noise reduction
    """
    
    def __init__(self):
        self.line_remover = StaffLineRemover()
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline"""
        # Step 1: Remove staff lines
        no_lines = self.line_remover.process(image)
        
        # Step 2: Enhance contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(no_lines)
        
        # Step 3: Slight blur to reduce noise
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Step 4: Final binary threshold for clean edges
        _, binary = cv2.threshold(denoised, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary


# CLI for testing
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python line_remover.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        sys.exit(1)
    
    print(f"Processing: {image_path}")
    print("=" * 60)
    
    # Remove lines
    remover = StaffLineRemover()
    result, lines = remover.remove_lines(image)
    
    # Save results
    output_path = image_path.replace('.png', '_no_lines.png')
    lines_path = image_path.replace('.png', '_lines_only.png')
    
    cv2.imwrite(output_path, result)
    cv2.imwrite(lines_path, lines)
    
    print(f"Saved: {output_path}")
    print(f"Lines mask: {lines_path}")
    
    # Count detected line pixels
    line_pixels = np.sum(lines == 255)
    total_pixels = lines.size
    print(f"\nLine coverage: {line_pixels/total_pixels:.2%}")
