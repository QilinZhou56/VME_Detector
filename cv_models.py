# Python In-built packages
from pathlib import Path
import PIL
from PIL import ImageDraw, ImageFont
from PIL import Image
# External packages
import streamlit as st
import torch

# Local Modules
import settings
import helper
import numpy as np

# Setting page layout
st.set_page_config(
    page_title="Marine Species Exploration",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Computer Vision of Marine Species, Protect Our Environment 🐟🪸🪼🦀️🌊⭐")

# Sidebar
st.sidebar.header("CV Datasets")

# Model Options
model_type = st.sidebar.radio(
    "Select Dataset", ['Select...', 'Healthy vs Bleached Corals', 'Style Transfer', 'Marine General', 'FathomNet', 'CoralNet', 'Zero Shot'])
confidence = float(st.sidebar.slider(
    "Select Detection Model Confidence", 25, 100, 40)) / 100

param_path = None
model_path = None
# Selecting Detection Or Segmentation
if model_type == 'Marine General':
  model_path = Path(settings.DETECTION_MODEL1)
elif model_type == 'FathomNet':
  model_path = Path(settings.DETECTION_MODEL2)
elif model_type == 'CoralNet':
  model_path = Path(settings.ClASSIFICATION_MODEL1)
elif model_type == 'Healthy vs Bleached Corals':
  model_select = st.sidebar.selectbox(
        "Choose a health status classification model...", settings.HEALTH_MODEL_DICT.keys())
  model_path = Path(settings.HEALTH_MODEL_DICT.get(model_select))
elif model_type == 'Style Transfer':
  source_param = st.sidebar.selectbox(
        "Choose a style...", settings.STYLE_DICT.keys())
  param_path = Path(settings.STYLE_DICT.get(source_param))


# Load Pre-trained ML Model
if model_type != 'Style Transfer' and model_type != 'Zero Shot':
    try:
        if model_type == 'Healthy vs Bleached Corals':
            if model_select != 'ViT' and model_select != 'YOLOv8':
                model = helper.load_custom_or_pretrained_model(model_path)
            elif model_select == 'ViT':
                model = helper.load_vit_model(model_path)
            else:
                model = helper.load_yolo_model(model_path)
        elif model_path is not None:
            model = helper.load_yolo_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
else:
        model = param_path

if model_type == 'Style Transfer':
    st_frame = st.empty()

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)
source_img = None

# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2)
    # Sidebar
    st.sidebar.header("Task Buttons after Image/Video Selection")
    with col1:
        try:
            if source_img is None and param_path is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
                video_file = open(settings.DEFAULT_VIDEO, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes, autoplay=True)
                st.caption("Beautiful Corals :rainbow:	:beach_with_umbrella: :cherry_blossom:")
            elif source_img is None and param_path is not None:
                st.markdown("""
                Click Random Generate for random image generation.

                """)
                if st.sidebar.button('Random Generate Images'):
                  # Display generated images using GAN
                  st_frame = st.empty()
                  uploaded_image = None
                  helper._display_generated_frames(param_path, uploaded_image, st_frame, upload=False, video=False)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
    with col2:
        if source_img is None and param_path is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
            sad_file  = open(settings.DEFAULT_SAD_VIDEO, 'rb')
            sad_bytes = sad_file.read()
            st.video(sad_bytes, autoplay=True)
            st.caption("Sad Corals :hospital: :desert: 	:man_with_probing_cane:")
        else:
          if model_type == 'FathomNet' or model_type == 'Marine General':
            if st.sidebar.button('Detect Species'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                if res:
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                else:
                    st.write("No classes detected with high enough confidence.")
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    #st.write(ex)
                    st.write("No image is uploaded yet!")

          if model_type == "CoralNet":
            if st.sidebar.button('Classify Coral Ecosystem Species'):
                res = model.predict(uploaded_image,
                                                  conf=confidence
                                                  )
                st.write("Classification Results:")
                if res:
                    for r in res:
                        for c in range(5): 
                            class_id = r.probs.top5[c]
                            confidence = r.probs.top5conf.numpy()[c]
                            class_name = r.names[class_id]
                            st.write(f"Class ID: {class_id}, Name: {class_name}, Confidence: {confidence:.2f}")

          if model_type == "Healthy vs Bleached Corals":
            if st.sidebar.button('Classify Coral Health Status'):
                if model_select == 'YOLOv8':
                    res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                    st.write("Classification Results:")
                    if res:
                        for r in res:
                            for c in range(2): 
                                class_id = r.probs.top5[c]
                                confidence = r.probs.top5conf.cpu().numpy()[c]
                                class_name = r.names[class_id]
                                st.write(f"Class ID: {class_id}, Name: {class_name}, Confidence: {confidence:.2f}")
                    else:
                        st.write("No classes detected with high enough confidence.")
                elif model_select != 'ViT':
                    # Convert the uploaded file to a NumPy array
                    uploaded_image_np = np.array(uploaded_image)

                    # Assuming custom model uses a different preprocessing
                    processed_image = helper.preprocess_image(uploaded_image_np, target_size=(150, 150)) 
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

                    # Displaying the results in Streamlit
                    st.write(f"Prediction: The coral is {coral_status}.")
                    st.write(f"Confidence: {confidence:.2%}")
                else:
                    uploaded_image_np = np.array(uploaded_image)
                    # Preprocess the image for the PyTorch model
                    processed_image = helper.preprocess_image_pytorch(uploaded_image_np, (224, 224))
                    device = next(model.parameters()).device
                    processed_image = processed_image.to(device)

                    # Make a prediction
                    with torch.no_grad():
                        output = model(processed_image)
                        logits = output.logits  # Access the logits from the output
                        probabilities = torch.nn.functional.softmax(logits, dim=1)
                        probability = probabilities[0][1].item()  # Assuming class '1' is 'Bleached'

                    # Determine coral status based on probability
                    threshold = 0.5
                    coral_status = "Bleached" if probability > threshold else "Healthy"
                    confidence = probability if coral_status == "Bleached" else 1 - probability

                    # Displaying the results in Streamlit
                    st.write(f"Prediction: The coral is {coral_status}.")
                    st.write(f"Confidence: {confidence:.2%}")

          if model_type == "Style Transfer":
            if st.sidebar.button('Generate Images'):
                # Display generated images using GAN
                st_frame = st.empty()
                if uploaded_image is not None:
                   helper._display_generated_frames(param_path, uploaded_image, st_frame, upload=True, video=False)
                else:
                    st.write("Please upload an image first to generate.")

          if model_type == "Zero Shot":
              st.write("Only support JPEG image for now")
              # Load model and processor
              processor, model = helper.load_owlvit_model()

              # Text input for user to enter descriptions
              descriptions = st.sidebar.text_input("Enter descriptions for detection, separated by commas",
                                                      "a photo of a fish, a photo of a coral")
              texts = [desc.strip() for desc in descriptions.split(',')]

              # Assuming 'uploaded_image' is defined and loaded elsewhere in the application
              results = helper.detect_with_owlvit(processor, model, uploaded_image, texts)
              draw = ImageDraw.Draw(uploaded_image)
              confidence_threshold = st.sidebar.slider('Confidence Threshold for Zero Shot', 0.0, 1.0, 0.5, key='conf_threshold_owlvit')
              for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                if score >= confidence_threshold:
                  box = [round(b, 2) for b in box.tolist()]
                  draw.rectangle(box, outline="red", width=3)
                  label_text = f"{texts[label]}: {round(score.item(), 3)}"
                  draw.text((box[0], box[1] - 10), label_text, fill="orange", font_size=15)

              st.image(uploaded_image, caption='Detected Image', use_column_width=True)

              # Display detailed results in an expander
              with st.expander("Detailed Detection Results"):
                for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                  if score >= confidence_threshold:
                    st.write(f"Detected {texts[label]} with confidence {round(score.item(), 3)} at location {box.tolist()}")

elif source_radio == settings.VIDEO:
    if model_type == 'Healthy vs Bleached Corals':
          size = (640, 640) if model_select == "YOLOv8" else (150, 150) if model_select != 'ViT' else (224, 224)
          helper.play_stored_video(confidence, model, size, yolov=model_select=="YOLOv8", pth=model_select=='ViT', Classify1=True, Classify2=False, Detect=False, Generate=False)

    elif model_type in ['FathomNet', 'Marine General']:
          helper.play_stored_video(confidence, model, (640, 640), yolov=True, pth=False, Classify1=False, Classify2=False, Detect=True, Generate=False)

    elif model_type == 'CoralNet':
          helper.play_stored_video(confidence, model, (640, 640), yolov=True, pth=False, Classify1=False, Classify2=True, Detect=False, Generate=False)

    elif model_type == "Style Transfer":
          helper.play_stored_video(confidence, model, (640, 640), yolov=True, pth=False, Classify1=False, Classify2=False, Detect=False, Generate=True)

elif source_radio == settings.YOUTUBE:
    button_context = "youtube"
    if model_type == 'Healthy vs Bleached Corals':
          size = (640, 640) if model_select == "YOLOv8" else (150, 150) if model_select != 'ViT' else (224, 224)
          helper.play_youtube_video(confidence, model, size, yolov=model_select=="YOLOv8", pth=model_select=='ViT', Classify1=True, Classify2=False, Detect=False, Generate=False)

    elif model_type in ['FathomNet', 'Marine General']:
          helper.play_youtube_video(confidence, model, (640, 640), yolov=True, pth=False, Classify1=False, Classify2=False, Detect=True, Generate=False)

    elif model_type == 'CoralNet':
          helper.play_youtube_video(confidence, model, (640, 640), yolov=True, pth=False, Classify1=False, Classify2=True, Detect=False, Generate=False)

    elif model_type == "Style Transfer":
          helper.play_youtube_video(confidence, model, (640, 640), yolov=True, pth=False, Classify1=False, Classify2=False, Detect=False, Generate=True)

else:
    st.error("Please select a valid source type!")
