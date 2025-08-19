import os
from PIL import Image

def load_images(input_directory):
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for filename in os.listdir(input_directory):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(os.path.join(input_directory, filename))
    
    return image_files