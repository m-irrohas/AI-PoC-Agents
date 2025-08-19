def save_output(output_file, recognized_text):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(recognized_text)