from pathlib import Path

file_path = Path(__file__).resolve()
ROOT = file_path.parent  # This is now the root directory

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
YOUTUBE = 'YouTube'
SOURCES_LIST = [IMAGE, VIDEO, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'Marine Species Detection OpenCV'/ 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'fish1.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'fish1_detected.png'


# Videos config
VIDEO_DIR = ROOT / 'Marine Species Detection OpenCV'/ 'videos'
DEFAULT_VIDEO = VIDEO_DIR / 'beautiful_coral.mp4'
DEFAULT_SAD_VIDEO = VIDEO_DIR  / 'sad_corals.mp4'
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
MODEL_DIR = ROOT / 'Marine Species Detection OpenCV'/ 'weights'
DETECTION_MODEL1 = MODEL_DIR /  'underwater_best.pt'
DETECTION_MODEL2 = MODEL_DIR / 'fathom_best.pt'
"""
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'
"""
ClASSIFICATION_MODEL1 = MODEL_DIR / 'coral_yolov8x_best.pt'
HEALTH_MODEL_DICT = {'ViT': MODEL_DIR / 'best_vit_healthy.pth', 'EfficientNet': MODEL_DIR / 'best_efficientnet_healthy.h5', 'YOLOv8': MODEL_DIR / 'healthy_best_yolov8x.pt',
'Customized': MODEL_DIR / 'healthy_custom_model.h5'}
STYLE_DICT = {
    'Shinkai': MODEL_DIR / 'Shinkai.h5',
    'Hosoda': MODEL_DIR / 'Hosoda.h5',
    'Paprika': MODEL_DIR / 'Paprika.h5',
   'Hayao': MODEL_DIR / 'Hayao.h5',
   'Pollution': MODEL_DIR / 'g_model_healthy_to_bleached_000010.h5', #'g_model_healthy_to_bleached.h5',
   'Recovery': MODEL_DIR / 'g_model_bleached_to_healthy000010.h5' #'g_model_bleached_to_healthy.h5'
}


