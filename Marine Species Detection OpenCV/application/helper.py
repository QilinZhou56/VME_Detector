from ultralytics import YOLO
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import cv2
from pytube import YouTube
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import img_to_array
import keras_contrib.layers
import numpy as np
from PIL import Image
from GAN.cartoon_models import cartoon_generator
from GAN.cyclegan import Generator
import os
import random
from matplotlib import pyplot as plt
import settings

@st.cache_data
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

@st.cache_data
def load_custom_or_pretrained_model(model_path):
    return keras_load_model(model_path)

@st.cache_data
def load_vit_model(model_path):
    model_name = "google/vit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(model_name, do_rescale=False)
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load state dict and adjust keys
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("classifier.1.", "classifier.")  # Adjust the key names
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

@st.cache_data
def load_owlvit_model():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    return processor, model

def detect_with_owlvit(processor, model, image, texts):
    inputs = processor(text=[texts], images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    target_sizes = torch.Tensor([image.size[::-1]])  # Image sizes for rescaling
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    
    return results[0]  # Return detection results for the first image

def load_image(image_input):
    if isinstance(image_input, Image.Image):
        image = tf.convert_to_tensor(np.array(image_input), dtype=tf.uint8)
        image = tf.image.decode_jpeg(tf.image.encode_jpeg(image), channels=3)
    else:
        image = tf.io.read_file(image_input)
        image = tf.image.decode_jpeg(image, channels=3)
    
    image = tf.image.resize(image, [256, 256])
    image = (image / 127.5) - 1  # Normalize images to [-1, 1]
    return image

def load_dataset(path, batch_size):
    dataset = tf.data.Dataset.list_files(path + '/*.jpg')  # Adjust pattern if needed
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def preprocess_image(image, target_size):
    # Resize the image
    resized_image = cv2.resize(image, target_size)
    # Normalize the pixel values to the range [0, 1]
    normalized_image = resized_image.astype("float") / 255.0
    # Convert the image to a NumPy array and add a batch dimension
    array_image = img_to_array(normalized_image)
    array_image = np.expand_dims(array_image, axis=0)
    return array_image


def preprocess_image_pytorch(image, target_size):
    # Define transformations
    transforms = Compose([
        Resize(target_size),  # Resize to the input size expected by the model
        ToTensor(),  # Convert image to tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (example values)
    ])

    image = Image.fromarray(image)
    # Apply transformations
    return transforms(image).unsqueeze(0)  # Add batch dimension


def _display_generated_frames(param_path, image, st_frame, upload, video):
    if image is not None:
      resized_image = image.resize((256, 256))
      resized_image = np.array(resized_image)

      if 'healthy' not in str(param_path):
        model = cartoon_generator(256)
        model.load_weights(param_path)

        # Ensure the input has the batch dimension:
        resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension
        generated_image = model.predict(resized_image)
        generated_image = np.clip(generated_image[0] * 255, 0, 255).astype(np.uint8)
        st_frame.image(generated_image, caption='Generated Image', use_column_width=True)
      else:
        model = Generator()
        model.load_weights(param_path)
        resized_image = load_image(image)
        resized_image = tf.expand_dims(resized_image, axis=0)
        prediction = model.predict(resized_image)
        prediction = np.clip(prediction[0] * 127.5 + 127.5, 0, 255).astype(np.uint8)

        if 'g_model_healthy_to_bleached_000010.h5' in str(param_path):
          st_frame.image(prediction, caption='Polluted', use_column_width=True)
        else:
          st_frame.image(prediction, caption='Recovered', use_column_width=True)

    if upload == False and video == False:
      # Load your datasets
      train_healthy_dataset = load_dataset('/content/drive/Shareddrives/Computer Vision Project/Dataset/Coral Health/healthy_corals', batch_size=1)
      train_bleached_dataset = load_dataset('/content/drive/Shareddrives/Computer Vision Project/Dataset/Coral Health/bleached_corals', batch_size=1)

      example_healthy = next(iter(train_healthy_dataset))
      example_bleached = next(iter(train_bleached_dataset))
      if 'healthy' not in str(param_path) and 'generator' not in str(param_path):
        model = cartoon_generator(256)
        model.load_weights(param_path)
        example_chosen = random.choice([example_healthy, example_bleached])
        resized_image = np.array(example_chosen)
        # Ensure the input has the batch dimension:
        generated_image = model.predict(resized_image)
        generated_image = np.clip(generated_image[0] * 255, 0, 255).astype(np.uint8)

        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(resized_image[0])
        ax[1].imshow(generated_image)
        ax[0].set_title("Original", fontdict={'fontsize': 16, 'fontweight': 'bold'})
        ax[1].set_title("Generated", fontdict={'fontsize': 16, 'fontweight': 'bold'})
        ax[0].axis("off")
        ax[1].axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

      else:
        model = Generator()
        model.load_weights(param_path)
        if 'g_model_healthy_to_bleached_000010.h5' in str(param_path):
          resized_image = np.array(example_healthy)
          prediction = model.predict(resized_image)
          prediction = (prediction[0] * 127.5 + 127.5).astype(np.uint8)
          img = (resized_image[0] * 127.5 + 127.5).astype(np.uint8)

          fig, ax = plt.subplots(1, 2, figsize=(10, 10))
          ax[0].imshow(img)
          ax[1].imshow(prediction)
          ax[0].set_title("Original", fontdict={'fontsize': 16, 'fontweight': 'bold'})
          ax[1].set_title("Polluted", fontdict={'fontsize': 16, 'fontweight': 'bold'})
          ax[0].axis("off")
          ax[1].axis("off")
          plt.tight_layout()
          st.pyplot(fig, use_container_width=False)
        else:
          resized_image = np.array(example_bleached)
          prediction = model.predict(resized_image)
          prediction = (prediction[0] * 127.5 + 127.5).astype(np.uint8)
          img = (resized_image[0] * 127.5 + 127.5).astype(np.uint8)

          fig, ax = plt.subplots(1, 2, figsize=(10, 10))
          ax[0].imshow(img)
          ax[1].imshow(prediction)
          ax[0].set_title("Original", fontdict={'fontsize': 16, 'fontweight': 'bold'})
          ax[1].set_title("Recovered", fontdict={'fontsize': 16, 'fontweight': 'bold'})
          ax[0].axis("off")
          ax[1].axis("off")
          plt.tight_layout()
          st.pyplot(fig, use_container_width=False)


def adjust_text_attributes(image, font_scale_base=0.75):
    """Dynamically adjust font scale and text color for visibility."""
    height, width = image.shape[:2]
    font_scale = font_scale_base * (width / 800)  # Adjust scale based on image width
    x, y = int(width * 0.05), int(height * 0.1)  # Position the text at 5% width and 10% height
    sampled_region = image[max(y-10, 0):y+10, max(x-10, 0):x+200]
    if np.mean(sampled_region) > 127:
        color = (0, 0, 0)
    else:
        color = (255, 255, 255)
    return (font_scale, (x, y), color)


def display_predictions(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale_base=0.75, thickness=2):
    """Draw text on an image and adjust for font scale and color automatically."""
    font_scale, org, color = adjust_text_attributes(image, font_scale_base)
    return cv2.putText(image.copy(), text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    

def _display_detected_frames(conf, model, st_frame, image, target_size, yolov, pth):
    if yolov and not pth:
        image = cv2.resize(image, target_size)
        
        # Convert image from BGR to RGB since CV2 uses BGR by default
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform inference on an image
        results = model(image,agnostic=True)

        # Extract bounding boxes, classes, names, and confidences
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()

        # Iterate through the results
        for box, cls, conf in zip(boxes, classes, confidences):
            # Ensure the box coordinates are integers
            x1, y1, x2, y2 = map(int, box)  # Convert each coordinate to int explicitly
            confidence = conf
            detected_class = cls
            name = names[int(cls)]  # Ensure the class index is used as an integer

            # Draw the rectangle on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            # Prepare the label text
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_pos = (x1, y1 - 10)

            # Check if there is enough space above the box; if not, place the label below the box
            if y1 - label_size[1] - 10 < 0:  # Check if the top space is not enough
                label_pos = (x1, y2 + label_size[1] + 10)  # Move label below the box

            # Place the label on the image
            cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with detections
        # Convert back to BGR for display in Streamlit
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st_frame.image(image_bgr,
                       caption='Detected Video',
                       use_column_width=True)

    elif yolov == False and pth == True:
        # Classify image using ViT model
        processed_image = preprocess_image_pytorch(image, target_size)
        device = next(model.parameters()).device
        processed_image = processed_image.to(device)
        with torch.no_grad():
            output = model(processed_image)
            logits = output.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            probability = probabilities[0][1].item()
        coral_status = "Bleached" if probability > 0.5 else "Healthy"
        confidence = probability if coral_status == "Bleached" else 1 - probability
        text = f"Prediction: {coral_status} (Confidence: {confidence:.2%})"
        image_with_text = display_predictions(image, text)
        st.image(image_with_text, caption="Coral Health Prediction", channels="BGR", use_column_width=True)
    else:
        # Basic model prediction
        processed_image = preprocess_image(image, target_size)
        predictions = model.predict(processed_image)
        probability = predictions[0][0]
        coral_status = "Bleached" if probability > 0.5 else "Healthy"
        confidence = probability if coral_status == "Bleached" else 1 - probability
        text = f"Prediction: {coral_status} (Confidence: {confidence:.2%})"
        image_with_text = display_predictions(image, text)
        st_frame.image(image_with_text, channels="BGR", use_column_width=True)


def play_youtube_video(conf, model, target_size, yolov, pth, Classify1, Classify2, Detect, Generate):
    source_youtube = st.sidebar.text_input("YouTube Video URL")
    if Classify1 or Classify2 or Detect:
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.get_highest_resolution()
            vid_cap = cv2.VideoCapture(stream.url)
            st_frame = st.empty()
            if (Classify1 and st.sidebar.button('Classify Coral Health Status')) or (Classify2 and st.sidebar.button('Classify Coral Ecosystem Species')) or (Detect and st.sidebar.button('Detect Species')):
              while vid_cap.isOpened():
                  success, image = vid_cap.read()
                  if success:
                      _display_detected_frames(conf,
                                                      model,
                                                      st_frame,
                                                      image,
                                                      target_size, yolov, pth)
                  else:
                      vid_cap.release()
                      break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        
    if Generate:
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.get_highest_resolution()
            vid_cap = cv2.VideoCapture(stream.url)
            st_frame = st.empty()
            if st.sidebar.button('Generate Images'):
              while vid_cap.isOpened():
                  success, image = vid_cap.read()
                  if success:
                      _display_generated_frames(model, Image.fromarray(image), st_frame, upload=False, video=True)
                  else:
                      vid_cap.release()
                      break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model, target_size, yolov, pth, Classify1, Classify2, Detect, Generate):
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    
    if video_bytes:
        st.video(video_bytes)

    if Classify1 or Classify2 or Detect:
        try:
            vid_cap = cv2.VideoCapture(
                    str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            if (Classify1 and st.sidebar.button('Classify Coral Health Status')) or (Classify2 and st.sidebar.button('Classify Coral Ecosystem Species')) or (Detect and st.sidebar.button('Detect Species')):
              while (vid_cap.isOpened()):
                  success, image = vid_cap.read()
                  if success:
                    _display_detected_frames(conf,
                                                      model,
                                                      st_frame, 
                                                      image, target_size, yolov, pth
                                                      )
                  else:
                      vid_cap.release()
                      break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

    if Generate:
        try:
            vid_cap = cv2.VideoCapture(
                    str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            if st.sidebar.button('Generate Images'):
              while (vid_cap.isOpened()):
                  success, image = vid_cap.read()
                  if success:
                      _display_generated_frames(model, Image.fromarray(image), st_frame, upload=False, video=True)
                  else:
                      vid_cap.release()
                      break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        