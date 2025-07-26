import os
import cv2
import tqdm
import numpy as np
from pathlib import Path
from ultralytics import SAM


# Define input and output directories
input_folder = "dbs/segmentor_db/raw"
output_folder = "dbs/segmentor_db/cooked"


# Initialize SAM model
model_path = "models/mobile_sam.pt"
if not os.path.exists(model_path):
    print(f"Model file {model_path} not found. Attempting to download...")

try:
    # Initialize SAM model
    model = SAM(model_path)  # Load MobileSAM model
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


for filename in os.listdir(input_folder):
    print(filename)
    image_path = os.path.join(input_folder, filename)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    print(1)
    # Convert BGR to RGB (SAM expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(2)
    # Perform segmentation
    results = model.predict(image_rgb, retina_masks=True, imgsz=1024)
    print(results)

    if results[0].masks is not None:
        masks = results[0].masks.data  # Shape: [num_masks, height, width]

        masks_np = masks.cpu().numpy()

        # Save NumPy array as .npz file
        output_filename = os.path.splitext(filename)[0] + ".npz"
        output_path = os.path.join(output_folder, output_filename)
        np.savez(output_path, masks=masks_np)
        print(f"Saved tensor to: {output_path}")
        
    else:
        print(f"No masks found for image: {image_path}")
