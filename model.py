import cv2
import numpy as np
from skimage import exposure
import easyocr
from collections import defaultdict
import json
import re

#initialize OCR_reader
OCR_reader = easyocr.Reader(['en'])

# Preparing the image
def preprocess_image(image):
    
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('grayscale_image.jpg', gray_image)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, h=30)
    cv2.imwrite('denoised_image.jpg', denoised_image)
    thresholded_image = cv2.adaptiveThreshold(
        denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite('thresholded_image.jpg', thresholded_image)
    contrast_enhanced_image = exposure.equalize_adapthist(thresholded_image, clip_limit=0.03)
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(contrast_enhanced_image, -1, sharpening_kernel)
    sharpened_image = (sharpened_image * 255).astype(np.uint8)
    cv2.imwrite('sharpened_image.jpg', sharpened_image)
    return sharpened_image


def extract_text_regions(image):
    ocr_results = OCR_reader.readtext(image, detail=1, paragraph=True)
    extracted_texts = []
    for result in ocr_results:
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

        text_height = y_max - y_min
        y_center = (y_min + y_max) // 2

        extracted_texts.append((text, y_center, text_height))
        print(f"Detected Text: '{text}', Y-Center: {y_center}, Height: {text_height}")

    return extracted_texts

def refine_headers_and_content(extracted_texts, header_height_threshold=80):
    organized_data = defaultdict(list)
    current_header = None
    extracted_texts.sort(key=lambda x: x[1])

    for text, y_center, height in extracted_texts:
        words = text.split()
        if re.match(r'^[A-Z][a-z]+', words[0]) or height > header_height_threshold:
            current_header = words[0]
            content_words = words[1:] if len(words) > 1 else []

            if current_header:
                organized_data[current_header] = []
            if content_words:
                organized_data[current_header].extend(content_words)

        elif current_header:
            organized_data[current_header].extend(words)

    organized_data = {
        header: ', '.join(re.sub(r'[;]+', '', word) for word in content)
        for header, content in organized_data.items()
    }

    return organized_data

def save_to_json(data, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
# main function
def main(image, output_file, header_height_threshold=100):
    preprocessed_image = preprocess_image(image)
    extracted_texts = extract_text_regions(preprocessed_image)
    organized_data = refine_headers_and_content(extracted_texts, header_height_threshold)
    save_to_json(organized_data, output_file)

    print("Text extraction and organization completed successfully.")
    return organized_data

#function calling 
if __name__ == "__main__":
    image = 'sample.jpeg' 
    output_file = 'extracted_text.json'
    organized_data = main(image, output_file)
    print(json.dumps(organized_data, indent=4))
