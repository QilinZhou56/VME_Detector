import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm

# Set the base directory for images
base_dir = './Coral Health'
full_path = os.path.abspath(base_dir)

# Sidebar for actions
st.sidebar.title("Coral Health Image Analysis🪸")
category = st.sidebar.selectbox("Select Image Category", ["healthy_corals", "bleached_corals", "all_categories"])
analyze_button = st.sidebar.button("Analyze Random Image")
eda_button = st.sidebar.button("Perform EDA on All Images")
def load_image_properties(category, seed=42):
    random.seed(seed)
    category_path = os.path.join(full_path, category)
    image_files = os.listdir(category_path)
    random_image_file = random.choice(image_files)
    image_path = os.path.join(category_path, random_image_file)
    try:
        with Image.open(image_path) as img:
            image_properties = {
                'file_name': random_image_file,
                'format': img.format,
                'size': img.size,
                'mode': img.mode,
                'path': image_path
            }
    except IOError:
        st.error(f"Error loading image {random_image_file}")
        image_properties = {}
    return image_properties

def display_image_properties(properties):
    st.write("### Sample Image Properties")
    updated = properties.copy()
    del updated['path']
    st.json(updated)

def display_image(image_path):
    img = Image.open(image_path)
    st.image(img, caption='Sample Image', use_column_width=True)

def plot_histogram(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    plt.figure(figsize=(7, 3.5))
    if img_array.ndim == 3:
        colors = ("red", "green", "blue")
        channel_labels = {0: 'Red Channel', 1: 'Green Channel', 2: 'Blue Channel'}
        for i, color in enumerate(colors):
            hist, bins = np.histogram(img_array[:,:,i], bins=256, range=[0,256])
            plt.plot(bins[:-1], hist, color=color, label=channel_labels[i])
    else:
        hist, bins = np.histogram(img_array, bins=256, range=[0, 256])
        plt.plot(bins[:-1], hist, color='gray', label='Intensity')
    plt.legend()
    plt.title('Histogram of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Define feature extraction and plotting functions
def extract_features(image_path):
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            features = {
                'mean_red': np.mean(img_array[:, :, 0]),
                'mean_green': np.mean(img_array[:, :, 1]),
                'mean_blue': np.mean(img_array[:, :, 2]),
                'std_red': np.std(img_array[:, :, 0]),
                'std_green': np.std(img_array[:, :, 1]),
                'std_blue': np.std(img_array[:, :, 2]),
                'width': img.width,
                'height': img.height
            }
            return features
    except IOError:
        st.error(f"Error processing image {image_path}")
        return None

def create_dataframe(base_dir, categories):
    data = []
    for category in categories:
        category_path = os.path.join(base_dir, category)
        image_files = os.listdir(category_path)
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            features = extract_features(image_path)
            if features:
                features['label'] = category
                data.append(features)
    return pd.DataFrame(data)

def plot_combined_histogram(dataframe, feature, title, xlabel, bins=30):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    for label in dataframe['label'].unique():
        subset = dataframe[dataframe['label'] == label]
        ax.hist(subset[feature], bins=bins, alpha=0.5, label=f'{label} {feature}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def plot_multiple_histograms(dataframe, features):
    """
    Plots multiple histograms for a list of features in a single figure.

    Args:
    dataframe (DataFrame): The pandas dataframe containing the data.
    features (list): A list of dictionaries specifying the features and plot details.
    """
    n_features = len(features)
    plt.figure(figsize=(15, 5 * n_features))
    count = 0
    for i, feature_info in enumerate(features, 1):
        plt.subplot(n_features, 1, i)
        plot_combined_histogram(dataframe, **feature_info) 
 
    plt.tight_layout()

st.title("Coral Health Image Analysis🪸")

# Main area
if analyze_button and category != "all_categories":
    properties = load_image_properties(category, seed=random.randint(1, 1000))
    display_image_properties(properties)
    display_image(properties['path'])
    plot_histogram(properties['path'])

if eda_button or (analyze_button and category == "all_categories"):
  categories = ["healthy_corals", "bleached_corals"]
  coral_health_df = create_dataframe(base_dir, categories)
  features_to_plot = [
      {'feature': 'mean_red', 'title': 'Histogram of Mean Red Values', 'xlabel': 'Mean Red Value'},
      {'feature': 'mean_green', 'title': 'Histogram of Mean Green Values', 'xlabel': 'Mean Green Value'},
      {'feature': 'mean_blue', 'title': 'Histogram of Mean Blue Values', 'xlabel': 'Mean Blue Value'},
      {'feature': 'std_red', 'title': 'Histogram of Red Standard Deviation', 'xlabel': 'Red Standard Deviation'},
      {'feature': 'std_green', 'title': 'Histogram of Green Standard Deviation', 'xlabel': 'Green Standard Deviation'},
      {'feature': 'std_blue', 'title': 'Histogram of Blue Standard Deviation', 'xlabel': 'Blue Standard Deviation'}
  ]

  plot_multiple_histograms(coral_health_df, features_to_plot)

if st.sidebar.button('Run Logistic Regression'):
    categories = ["healthy_corals", "bleached_corals"]
    coral_health_df = create_dataframe(base_dir, categories)
    st.write("Running Logistic Regression...")  # Debug statement

    label_mapping = {'healthy_corals': 0, 'bleached_corals': 1}
    coral_health_df['label_encoded'] = coral_health_df['label'].map(label_mapping)
    X = coral_health_df.drop(['label', 'label_encoded'], axis=1)
    y = coral_health_df['label_encoded']

    try:
        X_const = sm.add_constant(X)
        logit_model = sm.Logit(y, X_const)
        logit_result = logit_model.fit(disp=0)
        results_summary = logit_result.summary2().tables[1]
        results_df = pd.DataFrame(results_summary)
        st.dataframe(results_df)  # Display the DataFrame in Streamlit
    except Exception as e:
        st.error("An error occurred: {}".format(e))  # Display errors in Streamlit

else:
    st.write("Click the button to run the logistic regression.")
