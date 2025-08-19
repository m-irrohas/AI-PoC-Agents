# OCR Application

## Overview
This application allows users to upload images and extract text using OCR.

## Setup
1. Clone the repository.
2. Create a `.env` file with your database URL.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run the application using `python app.py`.

## API Endpoints
- `POST /api/upload`: Upload an image for OCR processing.
- `GET /api/results/<id>`: Retrieve the extracted text for a given image ID.