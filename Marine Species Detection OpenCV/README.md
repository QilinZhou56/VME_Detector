# Marine Species Detection OpenCV
# Referenced Tutorials
1. Youtube OpenCV channel ![TheCodingBug] (https://www.youtube.com/hashtag/thecodingbug).
2. MBARI Deep-Sea Guide ![website link](http://dsg.mbari.org/dsg/browsetreesearch/concept/marine%20organism)
3. Yolov-8 streamlit App Guide [GitHub link)(https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/tree/master)
   
# SSD 
SSD (Single Shot MultiBox Detector): This is the architecture used for the model. SSD is designed for object detection, which means it predicts both bounding boxes and class labels for each detected object in an image. It does this in a single forward pass of the network, making it efficient and fast.
# Models
1. Mobilenet
2. YOLOv8 
# Current Step:
1. Build a file of deep sea class names
2. Modify and train based on YOLOv8 and other CV models
3. We are working on tailoring the model specific to vulnerable corals, I am just creating this interface for different models' application on videos.
# Videos 
Deep Fish Datasets


## Training (YOLOv8)
1. Underwater Object Detection Dataset [Kaggle Link](https://www.kaggle.com/datasets/akshatsng/underwater-dataset-for-8-classes-with-label?select=valid)
   - Seven Classes: Fish, Crab, Human, Trash, Jellyfish, Coral Reef, Fish Group
   - Dataset Split: Train Set: 70%; Validation Set: 10%; Test Set: 20% (640x640)
2. Metrics
![Confusion Matrix](https://github.com/QilinZhou56/VME_Detector/blob/main/Marine%20Species%20Detection%20OpenCV/YOLOv8_metrics/confusion_matrix.png)
![F1 curve](https://github.com/QilinZhou56/VME_Detector/blob/main/Marine%20Species%20Detection%20OpenCV/YOLOv8_metrics/F1_curve.png)
![PR curve](https://github.com/QilinZhou56/VME_Detector/blob/main/Marine%20Species%20Detection%20OpenCV/YOLOv8_metrics/PR_curve.png)
3. Video Test Example
[![YouTube](http://i.ytimg.com/vi/W3hhkOUMFsY/hqdefault.jpg)](https://www.youtube.com/watch?v=W3hhkOUMFsY)
   
