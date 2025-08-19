import easyocr

def perform_ocr(image):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en', 'ja'])  # Support for English and Japanese
    results = reader.readtext(image)
    
    # Extract text from results
    extracted_text = ' '.join([result[1] for result in results])
    
    return extracted_text