import os
import pickle
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTModel
from tqdm import tqdm

# Function to process a single image and return its feature embedding and the image array
def process_image(image_path, processor, model):
    with Image.open(image_path) as img:
        img_array = np.array(img)  # Convert the image to a numpy array
        if img_array.dtype != np.uint8:  # Ensure the array is of type uint8
            img_array = img_array.astype(np.uint8)
        inputs = processor(img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].squeeze(0)   # (768,) shape
        return features.cpu().numpy(), img_array

# Initialize the model and processor
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

# Define the batch size and total image limit per folder
batch_size = 1000
total_images_limit = 5000

# Folder names to process
folder_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# Base directory containing image folders
base_dir = "D:\\CV_project\\NCT-CRC-HE-100K\\NCT-CRC-HE-100K"

# Loop through each folder and process images
for folder in folder_names:
    image_dir = os.path.join(base_dir, folder)
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')]
    image_files = image_files[:total_images_limit]  # Limit the number of images processed

    # Process images in batches
    for batch_index in range(0, len(image_files), batch_size):
        embeddings_list = []  # List to store image embeddings and image arrays for the current batch
        # Process each image in the current batch
        for i in tqdm(range(batch_index, min(batch_index + batch_size, len(image_files))), desc=f"Processing {folder} batch {batch_index//batch_size + 1}"):
            embeddings, image_array = process_image(image_files[i], image_processor, model)
            embeddings_list.append({'image_array': image_array, 'feat_patch': embeddings})

        # Save embeddings to a pickle file for the current batch
        pickle_filename = f"D:\\CV_project\\{folder}_image_embeddings_batch_{batch_index//batch_size + 1}.pkl"
        with open(pickle_filename, "wb") as f:
            pickle.dump(embeddings_list, f)
