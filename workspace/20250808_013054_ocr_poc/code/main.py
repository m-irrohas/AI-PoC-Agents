import os
import logging
from preprocessing import preprocess_image
from ocr_processing import perform_ocr
from post_processing import format_output

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    input_directory = './data'
    output_directory = './output'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each image in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_directory, filename)
            logging.info(f'Processing file: {image_path}')
            try:
                # Preprocess the image
                preprocessed_image = preprocess_image(image_path)
                
                # Perform OCR
                extracted_text = perform_ocr(preprocessed_image)
                
                # Format the output
                formatted_text = format_output(extracted_text)
                
                # Save the output to a text file
                output_file_path = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}.txt')
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(formatted_text)
                
                logging.info(f'Successfully processed: {filename} -> {output_file_path}')
            except Exception as e:
                logging.error(f'Error processing {filename}: {e}')

if __name__ == '__main__':
    main()