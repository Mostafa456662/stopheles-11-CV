import os
import cv2
import numpy as np
from pathlib import Path
import tqdm

# Define input and output directories
input_folder = "dbs/segmentor_db/raw"  # Input folder with original images
output_folder = "dbs/segmentor_db/cooked"  # Folder with saved .npz files

# Process each .npz file in the output folder with progress bar
for filename in tqdm.tqdm(os.listdir(output_folder), desc="Visualizing masks"):
    if not filename.endswith(".npz"):
        continue
    
    # Get corresponding image file
    image_filename = os.path.splitext(filename)[0] + ".png"  # Adjust extension if needed (e.g., .png)
    image_path = os.path.join(input_folder, image_filename)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    
    # Debug: Print image shape and type
    print(f"Processing {image_filename}: Image shape = {image.shape}, dtype = {image.dtype}")
    
    # Load masks from .npz file
    npz_path = os.path.join(output_folder, filename)
    try:
        data = np.load(npz_path)
        masks_np = data['masks']  # Shape: [num_masks, height, width]
        print(f"Loaded masks from {npz_path}: Masks shape = {masks_np.shape}, min = {masks_np.min()}, max = {masks_np.max()}")
    except Exception as e:
        print(f"Error loading .npz file {npz_path}: {e}")
        continue
    
    # Convert BGR to RGB for consistency
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a blank image for combined masks
    combined_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    
    # Generate random colors for each mask
    num_masks = masks_np.shape[0]
    colors = np.random.randint(50, 255, size=(num_masks, 3), dtype=np.uint8)  # Avoid very dark colors
    
    # Create overlay image starting with original RGB image
    overlay_image = image_rgb.copy()
    
    # Process each mask
    for i in range(num_masks):
        mask = masks_np[i].astype(np.uint8)  # Convert to uint8 (0 or 1), shape: [height, width]
        
        # Create 2D boolean mask for indexing
        mask_area = mask > 0  # Shape: [height, width]
        
        # Create colored mask with 3 channels
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        colored_mask[mask_area] = colors[i]  # Apply color to mask area
        
        # Add to combined mask image
        combined_mask = np.maximum(combined_mask, colored_mask)
        
        # Overlay mask on image with transparency (50% image, 50% mask)
        overlay_image[mask_area] = (0.5 * overlay_image[mask_area] + 0.5 * colored_mask[mask_area]).astype(np.uint8)
    
    # Convert images to BGR for OpenCV display
    overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
    combined_mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_RGB2BGR)
    
    # Debug: Check overlay and mask image stats
    print(f"Overlay image: min = {overlay_image_bgr.min()}, max = {overlay_image_bgr.max()}")
    print(f"Combined mask: min = {combined_mask_bgr.min()}, max = {combined_mask_bgr.max()}")
    
    # Display original image with overlaid masks
    cv2.imshow(f"Image with Masks: {image_filename}", overlay_image_bgr)
    # Display combined masks
    cv2.imshow(f"Masks: {image_filename}", combined_mask_bgr)
    
    # Wait for key press; press 'q' to quit, any other key to continue
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()