# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Marine Species Detection using YOLOv8",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Marine Species Detection using YOLOv8, Protect Our Environment🪸")

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
    model_path = Path(settings.ClASSIFICATION_MODEL2)
elif model_type == 'Style Transfer':
    source_param = st.sidebar.selectbox(
        "Choose a style...", settings.STYLE_DICT.keys())
    param_path = Path(settings.STYLE_DICT.get(source_param))

# Load Pre-trained ML Model
if model_type != 'Style Transfer':
    try:
        model = helper.load_model(model_path)
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

            if st.sidebar.button('Generate Images'):
                # Display generated images using GAN
                st_frame = st.empty()
                if uploaded_image is not None:
                   helper._display_generated_frames(param_path, uploaded_image, st_frame)
                else:
                    st.write("Please upload an image first to generate.")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")