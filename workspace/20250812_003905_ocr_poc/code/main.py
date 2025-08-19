import os
import time
import json
import logging
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import easyocr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
INPUT_DIR = './data/sample_images'
OUTPUT_DIR = './data/results'
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'results.json')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

def process_image(image_path):
    """Process a single image to extract text."""
    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)

        # Perform OCR
        start_time = time.time()
        results = reader.readtext(img_array)
        processing_time = time.time() - start_time

        # Extract text and confidence
        extracted_text = " ".join([result[1] for result in results])
        confidence_scores = [result[2] for result in results]

        return extracted_text, confidence_scores, processing_time

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None, None, None

def save_results(results):
    """Save results to a JSON file."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)

def visualize_results(image_path, extracted_text):
    """Visualize the results by overlaying the extracted text on the image."""
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f'Extracted Text: {extracted_text}')
    plt.axis('off')
    plt.show()

def main():
    """Main function to process images in the input directory."""
    results = []
    
    # Process each image in the input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(INPUT_DIR, filename)
            logging.info(f"Processing image: {image_path}")

            extracted_text, confidence_scores, processing_time = process_image(image_path)

            if extracted_text is not None:
                results.append({
                    'filename': filename,
                    'extracted_text': extracted_text,
                    'confidence_scores': confidence_scores,
                    'processing_time': processing_time
                })
                visualize_results(image_path, extracted_text)

    # Save results to JSON
    save_results(results)
    logging.info(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()