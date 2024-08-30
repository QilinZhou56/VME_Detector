# Reef Madness Project
![Enjoy our Reef Madness](https://github.com/QilinZhou56/VME_Detector/blob/main/ReefMadness.png)
## Team Members:
Jaskirat, Grey, Qilin (CAPPies)

## Related Repository, Report and Presentation
Our primary repository is [here](https://github.com/jaskcodes/Reef-madness/tree/main/detection_opencv_streamlit/application). We use **this only to connect with streamlit** with trained weights files and image repositories. 

You can find the project report [here](https://github.com/QilinZhou56/VME_Detector/blob/main/project_report.pdf) and the final presentation [here](https://github.com/QilinZhou56/VME_Detector/blob/main/presentation.pdf).

## 

## Application Use 
```bash
streamlit run cv_models.py
```
## Application Use 
Users should download weights at [here](https://drive.google.com/drive/folders/1CaCDrzdqK_Nx_ZMj6eVXXsDuwv6nm9bz?usp=drive_link). They could also download sample video files at [here](https://drive.google.com/drive/folders/1GGrGFj3aGKB7I294s7Y-GpIbd2vm9kZ1?usp=drive_link). These files should be placed under "Marine Species Detection OpenCV" directory. For any further emerging problems, please email at [qilinzhou888@gmail.com](mailto:qilinzhou888@gmail.com). 

## Project Overview
![schema](Marine%20Species%20Detection%20OpenCV/schema.png)
Climate change poses significant threats to our marine ecosystems, particularly to coral reefs, many of which are experiencing bleaching. This project aims to address this issue using deep learning techniques, divided into two parts: 

1. **Classification of Coral Health Status**
2. **Generative Models to Visualize Potential Unhealthy States of Currently Healthy Corals**

## Part 1: Classification of Coral Health Status

- **Objective**: Identify healthy and unhealthy corals and classify coral species.
- **Models Used**: 
  - Custom
  - YOLO8
  - EfficientNet
  - Vision Transformer

- **Performance**: 
  - The Vision Transformer model achieved the best performance.
  - Loss: 0.32
  - Test Accuracy: 85.87%
  - Dataset: 923 images

- **Species Classification**:
  - Applied YOLO8 with the Coral Net dataset.
  - Enabled the user interface to display species types on the web application.

## Part 2: Generative Models

- **Objective**: Visualize how unhealthy coral reefs would look if they were healthy and vice versa.
- **Models Developed**: Custom CycleGANs
- **Pre-trained Cartoon GANs**:
  - Inspired by directors like Hayao Miyazaki, Mamoru Hosoda, Satoshi Kon, and Makoto Shinkai.
  - Transformed images into specific artistic styles.
  - Offered a unique way to explore ocean species in digital media.

## Additional Analysis on Detection Model

- **Dataset**: FathomNet (includes labeled bounding boxes).
- **Species**: Focused on seven species due to limited computational resources and time.
- **Challenges**: Custom model faced challenges in achieving high accuracy.
- **Solution**: Applied YOLO8 directly to demonstrate object detection on the web app.

## Interactive Streamlit App

- **Features**:
  - Users can upload images, and videos, or choose from YouTube.
  - Perform classification, apply GAN transformations, or detect species bounding boxes.
  - Utilizes different models on various datasets.

## Team Contributions

- **Grey**:
  - Focused on EDA and model selection for the healthy coral datasets.
  - Compared models and managed the ML flow pipeline for classification.

- **Jaskirat**:
  - Worked on EDA and detection problems with the FathomNet dataset.
  - Fine-tuned the Vision Transformer.
  - Generated evaluation metrics and tracked experiments on MLFlow and Databricks.

- **Qilin**:
  - Customized and deployed the GAN model.
  - OWL-ViT and Yolov8 Detection Model Exploration.
  - Built the web application for the project.

- **Collaborative Efforts**:
  - Worked together on the slides and the report.
    
## Directory 
```plaintext
├── GAN
├── cv_models.py
├── settings.py
├── helper.py
├── pages
├── requirements.txt
├── Marine Species Detection Open CV (initial web attempt)
│   ├── videos
│   ├── images
│   ├── weights
│   ├── application (demo in Google Colab, demo for detection on cpu)
│   ├── Coral Health (dataset)
└── .gitignore
```
**Note:** all streamlit app related py files are moved under VME_Detector directory, to be compliant with streamlit cloud required format.


