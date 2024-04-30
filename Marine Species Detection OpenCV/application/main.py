from Detector_Marine_on_cpu import *
import settings
from pathlib import Path

def main():
    videoPath = "test_videos/fish3.mp4"
    """
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath  = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()
    """
    modelPath1 = Path(settings.DETECTION_MODEL1)
    modelPath2 = Path(settings.DETECTION_MODEL2)
    detector = Detector(videoPath,  modelPath1=modelPath1, modelPath2=modelPath2)
    detector.onVideo()

if __name__ == "__main__":
    main()
