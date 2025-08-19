import os
import logging
from input_module import load_images
from ocr_module import perform_ocr
from output_module import save_output

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    input_directory = './data'
    output_directory = './output'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    logging.info("Loading images from directory: %s", input_directory)
    images = load_images(input_directory)

    for image_path in images:
        try:
            logging.info("Processing image: %s", image_path)
            recognized_text = perform_ocr(image_path)
            output_file = os.path.join(output_directory, os.path.basename(image_path) + '.txt')
            save_output(output_file, recognized_text)
            logging.info("Saved output to: %s", output_file)
        except Exception as e:
            logging.error("Error processing image %s: %s", image_path, str(e))

if __name__ == "__main__":
    main()