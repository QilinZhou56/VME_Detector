import streamlit as st
import requests
import plotly.express as px
from PIL import Image
import pandas as pd
from io import BytesIO
import pydeck as pdk

# FathomNet API interactions and other utilities
from fathomnet.api import boundingboxes, images

# Setting page layout
st.set_page_config(
    page_title="Species Map",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Find the species of your interest 🪸")

# Make a bar chart of the top N concepts by bounding boxes
N = st.slider("Select number of top concepts to display", 5, 20, 11)

# Get the number of bounding boxes for all concepts
concept_counts = boundingboxes.count_total_by_concept()

# Sort by number of bounding boxes and get the top N concepts
sorted_concepts = sorted(concept_counts, key=lambda cc: cc.count, reverse=True)[:N]
concepts = [cc.concept for cc in sorted_concepts]
counts = [cc.count for cc in sorted_concepts]

# Display the bar chart
fig = px.bar(
    x=concepts,
    y=counts,
    labels={'x': 'Concept', 'y': 'Bounding box count'},
    title=f'Top {N} concepts',
    text_auto=True,
    color=concepts 
)
st.plotly_chart(fig)

# Sidebar interaction for concept selection
all_concepts = [cc.concept for cc in concept_counts]
selected_concept = st.sidebar.selectbox(f"Choose a concept to explore ({len(concept_counts)} in total)", all_concepts, index=1)

# Displaying images for the selected concept
concept_images = images.find_by_concept(selected_concept)
st.sidebar.write(f'Found {len(concept_images)} images of {selected_concept}')

# Extract the depth (in meters) from each image and show histogram
depths = [img.depthMeters for img in concept_images if img.depthMeters is not None]
depth_fig = px.histogram(
    y=depths,
    title=f'{selected_concept} images by depth',
    labels={'y': 'Depth (m)'},
    orientation='h'
)
depth_fig['layout']['yaxis']['autorange'] = 'reversed'
st.plotly_chart(depth_fig)

# Data preparation
locations = [(img.latitude, img.longitude) for img in concept_images if img.latitude is not None and img.longitude is not None]
df_locations = pd.DataFrame(locations, columns=['lat', 'lon'])

# Define a layer to use on the map
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=df_locations,
    opacity=0.9,
    get_position=["lon", "lat"],
    threshold=0.3,
    get_weight=1
)

# Set the viewport location
view_state = pdk.ViewState(latitude=36.807, longitude=-121.988, zoom=10, bearing=0, pitch=0)

# Render the map
r = pdk.Deck(
    layers=[heatmap_layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/light-v9', 
)
st.pydeck_chart(r)

if selected_concept != None:
    # Optionally, show thumbnails of some images
    if st.sidebar.checkbox("Show thumbnails of images"):
        cols = st.columns(4)  # adjust number of columns based on layout preference
        for index, img in enumerate(concept_images):
            if index < len(cols):  # Just showing a few images to fit in the grid
                with cols[index % 4]:
                    img_data = requests.get(img.url).content
                    try:
                        img_io = Image.open(BytesIO(img_data))
                        st.image(img_io, caption=f"{selected_concept} {index+1}")
                    except:
                        st.write("Image not public yet!")
