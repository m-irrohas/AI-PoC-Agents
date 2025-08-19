import easyocr
import logging

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

def perform_ocr(image_path):
    logging.info("Performing OCR on: %s", image_path)
    result = reader.readtext(image_path)
    recognized_text = "\n".join([text[1] for text in result])
    return recognized_text