from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from GAN.cartoon_models import cartoon_generator

import settings

def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

def load_custom_model(model_path):
    return keras_load_model(model_path)

def preprocess_image(image, target_size):
    # Resize the image
    resized_image = cv2.resize(image, target_size)
    # Normalize the pixel values to the range [0, 1]
    normalized_image = resized_image.astype("float") / 255.0
    # Convert the image to a NumPy array and add a batch dimension
    array_image = img_to_array(normalized_image)
    array_image = np.expand_dims(array_image, axis=0)
    return array_image

def _display_generated_frames(param_path, image, st_frame):
    resized_image = image.resize((256, 256))
    resized_image = np.array(resized_image)
    model = cartoon_generator(256)
    model.load_weights(param_path)
    # Ensure the input has the batch dimension:
    resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension
    generated_image = model.predict(resized_image)
    generated_image = np.clip(generated_image[0] * 255, 0, 255).astype(np.uint8)
    st_frame.image(generated_image, caption='Generated Image', use_column_width=True)


def _display_detected_frames(conf, model, st_frame, image, target_size, yolov):
    if yolov == True:
        # Resize the image to a standard size
        image = cv2.resize(image, target_size)
        
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf, agnostic_nms=True)
        
        # Plot the detected objects on the video frame
        res_plotted = res[0].plot()
        st_frame.image(res_plotted,
                    caption='Detected Video',
                    channels="BGR",
                    use_column_width=True
                    )
    else:
        processed_image = preprocess_image(image, target_size)
        predictions = model.predict(processed_image)
        # Determine class based on threshold (usually 0.5 for binary classifiers)
        threshold = 0.5
        probability = predictions[0][0]
        if probability > threshold:
            coral_status = "Bleached"
            confidence = probability
        else:
            coral_status = "Healthy"
            confidence = 1 - probability

        # Prepare the text to overlay
        text = f"Prediction: {coral_status} (Confidence: {confidence:.2%})"

        # Draw the text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (100, 100)  # Bottom-left corner of the text string in the image
        font_scale = 2
        color = (255, 0, 0) if coral_status == "Bleached" else (0, 255, 0)
        thickness = 4
        image_with_text = cv2.putText(image.copy(), text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        # Display the image with the text in Streamlit
        st_frame.image(image_with_text, channels="BGR", use_column_width=True)

def play_youtube_video(conf, model, target_size, yolov=True):
    source_youtube = st.sidebar.text_input("YouTube Video URL")
    if st.sidebar.button('Classify Coral Health Status') or st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.get_highest_resolution()
            vid_cap = cv2.VideoCapture(stream.url)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                                    model,
                                                    st_frame,
                                                    image,
                                                    target_size, yolov)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        
    if st.sidebar.button('Generate Images'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.get_highest_resolution()
            vid_cap = cv2.VideoCapture(stream.url)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_generated_frames(model, Image.fromarray(image), st_frame) 
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_stored_video(conf, model, target_size, yolov):
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Classify Coral Health Status') or st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                    str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                                    model,
                                                    st_frame, 
                                                    image, target_size, yolov
                                                    )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

    if st.sidebar.button('Generate Images'):
        try:
            vid_cap = cv2.VideoCapture(
                    str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_generated_frames(model, Image.fromarray(image), st_frame)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        