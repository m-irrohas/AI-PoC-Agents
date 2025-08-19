import os
import json
import time
import logging
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
INPUT_DIR = './data/sample_images'
OUTPUT_FILE = 'output/results.json'
RESULTS_DIR = 'output'
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

def preprocess_image(image_path):
    """Preprocess the image for OCR."""
    try:
        # Load image
        image = cv2.imread(image_path)
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # Thresholding
        _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_image
    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {e}")
        return None

def extract_text(image):
    """Extract text from the preprocessed image using Tesseract OCR."""
    try:
        text = pytesseract.image_to_string(image, lang='eng')
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return ""

def process_images():
    """Process all images in the input directory."""
    results = []
    start_time = time.time()

    for filename in os.listdir(INPUT_DIR):
        if any(filename.endswith(ext) for ext in IMAGE_FORMATS):
            image_path = os.path.join(INPUT_DIR, filename)
            logging.info(f"Processing image: {image_path}")

            # Preprocess image
            preprocessed_image = preprocess_image(image_path)
            if preprocessed_image is not None:
                # Extract text
                extracted_text = extract_text(preprocessed_image)
                results.append({
                    'filename': filename,
                    'extracted_text': extracted_text
                })
            else:
                results.append({
                    'filename': filename,
                    'extracted_text': None
                })

    end_time = time.time()
    processing_time = end_time - start_time
    logging.info(f"Processing completed in {processing_time:.2f} seconds.")

    # Save results to JSON
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logging.info(f"Results saved to {OUTPUT_FILE}")

def visualize_results():
    """Visualize the results of the OCR processing."""
    with open(OUTPUT_FILE, 'r') as f:
        results = json.load(f)

    for result in results:
        plt.figure()
        plt.title(result['filename'])
        plt.axis('off')
        plt.text(0.5, 0.5, result['extracted_text'] if result['extracted_text'] else "No text extracted", 
                 fontsize=12, ha='center', va='center')
        plt.show()

if __name__ == "__main__":
    process_images()
    visualize_results()