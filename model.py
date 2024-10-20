import cv2
import numpy as np
from skimage import exposure
import easyocr
from collections import defaultdict
import json
import re

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_image(image_path):
    """
    Preprocess the image to enhance OCR accuracy.
    """
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Enhance contrast
    enhanced = exposure.equalize_adapthist(thresh, clip_limit=0.03)

    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Convert to uint8
    sharpened = (sharpened * 255).astype(np.uint8)

    return sharpened

def extract_text_regions(image):
    """
    Use EasyOCR to detect text regions and extract text.
    """
    results = reader.readtext(image, detail=1, paragraph=True)
    extracted_texts = []

    for result in results:
        if len(result) == 3:
            bbox, text, _ = result
        elif len(result) == 2:
            bbox, text = result
        else:
            continue

        x_min = int(min([point[0] for point in bbox]))
        y_min = int(min([point[1] for point in bbox]))
        x_max = int(max([point[0] for point in bbox]))
        y_max = int(max([point[1] for point in bbox]))

        height = y_max - y_min
        y_center = (y_min + y_max) // 2

        extracted_texts.append((text, y_center, height))
        print(f"Detected Text: '{text}', Y-Center: {y_center}, Height: {height}")

    return extracted_texts

def refine_headers_and_content(extracted_texts, header_height_threshold=80):
    """
    Refine headers and content separation using improved logic.
    """
    organized_data = defaultdict(list)
    current_header = None

    # Sort extracted texts by vertical position
    extracted_texts = sorted(extracted_texts, key=lambda x: x[1])

    for text, y_center, height in extracted_texts:
        # Split text into words for better header detection
        words = text.split()

        # Identify headers and content
        if re.match(r'^[A-Z][a-z]+', words[0]) or height > header_height_threshold:
            current_header = words[0]
            content_words = words[1:] if len(words) > 1 else []
            if current_header:
                organized_data[current_header] = []
            if content_words:
                organized_data[current_header].extend(content_words)
        elif current_header:
            
            organized_data[current_header].extend(words)

    # Clean up content strings by removing unwanted punctuation
    organized_data = {header: ', '.join(re.sub(r'[;]+', '', word) for word in content)
                      for header, content in organized_data.items()}

    return organized_data

def save_to_json(data, output_path):
    """
    Save the organized data to a JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main(image_path, output_json_path, header_height_threshold=100):
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Step 2: Extract text regions using EasyOCR
    extracted_texts = extract_text_regions(preprocessed_image)

    # Step 3: Refine headers and content separation
    organized_data = refine_headers_and_content(
        extracted_texts, header_height_threshold=header_height_threshold
    )

    # Step 4: Save the organized data to JSON
    save_to_json(organized_data, output_json_path)

    print("Text extraction and organization completed successfully.")
    return organized_data

# Example usage
if __name__ == "__main__":
    image_path = 'sample.jpeg' 
    output_json_path = 'extracted_text.json'  

    # Run the complete model
    organized_data = main(image_path, output_json_path)

    # Display the organized output
    print(json.dumps(organized_data, indent=4))
