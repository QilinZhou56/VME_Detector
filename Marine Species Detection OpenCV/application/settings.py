from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
YOUTUBE = 'YouTube'
SOURCES_LIST = [IMAGE, VIDEO, YOUTUBE]

# Images config

IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'fish1.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'fish1_detected.png'


# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'fish1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'fish2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'fish3.mp4'
VIDEO_4_PATH = VIDEO_DIR / 'fish4.mp4'
VIDEO_5_PATH = VIDEO_DIR / 'fish5.mp4'
VIDEO_6_PATH = VIDEO_DIR / 'fish6.mp4'
VIDEO_7_PATH = VIDEO_DIR / 'coral1.mp4'
VIDEO_8_PATH = VIDEO_DIR / 'coral2.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,  
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
    'video_4': VIDEO_4_PATH,
    'video_5': VIDEO_5_PATH,
    'video_6': VIDEO_6_PATH,
    'video_7': VIDEO_7_PATH,
    'video_8': VIDEO_8_PATH,

}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL1 = MODEL_DIR /  'best.pt'
DETECTION_MODEL2 = MODEL_DIR / 'fathom_best.pt'
"""
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'
"""
ClASSIFICATION_MODEL1 = MODEL_DIR / 'yolov8n.pt'
HEALTH_MODEL_DICT = {'YOLOv8': MODEL_DIR / 'healthy_yolov8.pt',
'Customized': MODEL_DIR / 'healthy_custom_model.h5'}
STYLE_DICT = {
    'Shinkai': MODEL_DIR / 'Shinkai.h5',
    'Hosoda': MODEL_DIR / 'Hosoda.h5',
    'Paprika': MODEL_DIR / 'Paprika.h5',
   'Hayao': MODEL_DIR / 'Hayao.h5',
}


