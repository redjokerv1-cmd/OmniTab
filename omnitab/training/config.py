"""
Training Configuration - External paths and settings
"""

from pathlib import Path

# ========================================
# Learning Data Vault Configuration
# ========================================

# 외부 학습 데이터 저장소 경로
LEARNING_DATA_VAULT = Path("G:/Study/AI/learning-data-vault")

# 프로젝트별 경로
OMNITAB_DATA_PATH = LEARNING_DATA_VAULT / "omnitab"

# ========================================
# Training Settings
# ========================================

# YOLO 클래스 정의
YOLO_CLASSES = [str(i) for i in range(25)] + ['h', 'p', 'x', 'harmonic']
NUM_CLASSES = len(YOLO_CLASSES)

# 기본 훈련 설정
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMG_SIZE = 640


def get_data_manager():
    """Get LearningDataManager with correct path"""
    from .learning_data_manager import LearningDataManager
    return LearningDataManager(base_dir=str(OMNITAB_DATA_PATH))
