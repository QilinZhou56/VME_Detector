# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

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
st.title("Computer Vision of Marine Species, Protect Our Environment🪸")

# Sidebar
st.sidebar.header("CV Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task and Click Associated Buttons (Detect; Classify; Generate)", ['Ocean Species Detection', 'Specific Species Detection', 'Coral Reef Classification', 'Coral Healthy Status', 'Style Transfer'])
confidence = float(st.sidebar.slider(
    "Select Detection Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Ocean Species Detection':
    model_path = Path(settings.DETECTION_MODEL1)
elif model_type == 'Specific Species Detection':
    model_path = Path(settings.DETECTION_MODEL2)
elif model_type == 'Coral Reef Classification':
    model_path = Path(settings.ClASSIFICATION_MODEL1)
elif model_type == 'Coral Healthy Status':
    model_select = st.sidebar.selectbox(
        "Choose a health status classification model...", settings.HEALTH_MODEL_DICT.keys())
    model_path = Path(settings.HEALTH_MODEL_DICT.get(model_select))
elif model_type == 'Style Transfer':
    source_param = st.sidebar.selectbox(
        "Choose a style...", settings.STYLE_DICT.keys())
    param_path = Path(settings.STYLE_DICT.get(source_param))

# Load Pre-trained ML Model
if model_type != 'Style Transfer':
    try:
        if model_type == 'Coral Healthy Status':
            if model_select == 'Customized':
                model = helper.load_custom_model(model_path)
            else:
                model = helper.load_yolo_model(model_path)
        else:
            model = helper.load_yolo_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
else:
        model = param_path

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)
source_img = None

# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
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
                                confidence = r.probs.top5conf.numpy()[c]
                                class_name = r.names[class_id]
                                st.write(f"Class ID: {class_id}, Name: {class_name}, Confidence: {confidence:.2f}")
                    else:
                        st.write("No classes detected with high enough confidence.")
                elif model_select == 'Customized':
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

            if st.sidebar.button('Generate Images'):
                # Display generated images using GAN
                st_frame = st.empty()
                if uploaded_image is not None:
                   helper._display_generated_frames(param_path, uploaded_image, st_frame)
                else:
                    st.write("Please upload an image first to generate.")

elif source_radio == settings.VIDEO:
    if model_type == 'Coral Healthy Status':
        if model_select == "YOLOv8":
            helper.play_stored_video(confidence, model, (640, 640), yolov=True)
        elif model_select == "Customized":
            helper.play_stored_video(confidence, model, (150, 150), yolov=False)
    else:
        helper.play_stored_video(confidence, model, (640, 640), yolov=True)

elif source_radio == settings.YOUTUBE:
    if model_type == 'Coral Healthy Status':
        if model_select == "YOLOv8":
            helper.play_youtube_video(confidence, model, (640, 640), yolov=True)
        elif model_select == "Customized":
            helper.play_youtube_video(confidence, model, (150, 150), yolov=False)
    else:
        helper.play_youtube_video(confidence, model, (640, 640), yolov=True)

else:
    st.error("Please select a valid source type!")