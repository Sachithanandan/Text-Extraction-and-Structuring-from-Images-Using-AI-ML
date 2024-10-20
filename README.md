**Text Extraction and Structuring from Images Using AI/ML**

## **Overview**
This project uses an AI/ML-driven approach to extract, organize, and structure text from images. The solution integrates Optical Character Recognition (OCR), image preprocessing, and Natural Language Processing (NLP) techniques to separate headers and content into an organized dictionary format.

## **Solution Components**
The project combines AI-based OCR with heuristic-based logic to differentiate headers from content. Here’s how it works:

1. **Image Preprocessing**
   - The input image undergoes several preprocessing steps to enhance the quality of text detection:
     - **Grayscale Conversion**: Converts the image to grayscale for better contrast.
     - **Denoising**: Reduces noise to improve OCR accuracy.
     - **Thresholding**: Applies adaptive thresholding to highlight text regions.
     - **Sharpening**: Enhances the sharpness of text edges.
   - These steps are performed using **OpenCV** and **skimage** libraries.

2. **Optical Character Recognition (OCR) with EasyOCR**
   - The preprocessed image is passed to **EasyOCR**, a deep learning-based OCR model that extracts text from the image.
   - EasyOCR uses pre-trained neural network models to accurately detect text blocks, regardless of orientation, font style, or size.

3. **Text Structuring and Organization**
   - The extracted text blocks are analyzed based on their position, size, and formatting.
   - **Headers and content** are differentiated using a combination of the following heuristics:
     - **Font Size**: Larger font sizes are treated as potential headers.
     - **Positioning**: Vertical distance between text blocks helps determine the relationship between headers and content.
     - **Text Pattern Recognition**: AI/ML-inspired rules are used to identify headers and content based on linguistic cues (e.g., keywords, capitalization).
   - The final output is an **organized dictionary** where headers are keys, and associated content is the value.

## **Setup and Installation**

### **Prerequisites**
- Python 3.7+
- Required libraries: 
  - OpenCV
  - NumPy
  - EasyOCR
  - skimage
  - Craft-Text-Detector
  - Any other library mentioned in the code

### **Installation Steps**
1. **Clone the Repository**
   ```bash
   git clone <repository-link>
   cd <repository-directory>
   ```
2. **Install Required Libraries**
   Install all the dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` should contain:
   ```
   opencv-python
   numpy
   easyocr
   scikit-image
   craft-text-detector
   ```

3. **Run the Code**
   Provide the input image path and output JSON path in the code or through the command line:
   ```bash
   python main.py
   ```
   - **Input**: An image containing headings and subheadings.
   - **Output**: A JSON file containing structured headers and content.

## **Approach Breakdown**

### **1. Image Preprocessing**
   - The image is preprocessed to improve the visibility of text, using denoising, thresholding, and sharpening techniques.
   - Preprocessing is crucial for better OCR performance.

### **2. Text Detection with EasyOCR**
   - **EasyOCR**, a deep learning-based OCR library, is used for detecting and recognizing text in the image.
   - It uses convolutional neural networks (CNNs) to recognize text and works effectively on images with varied font styles and orientations.

### **3. Header and Content Separation**
   - AI/ML-inspired rules determine headers and content:
     - **Font size and position** serve as key indicators of headers.
     - NLP-inspired pattern recognition rules identify content lines and match them with appropriate headers.
     - The output is structured into a dictionary format.

### **4. Final Output**
   - The organized output is saved in a JSON file, where headers act as dictionary keys, and the associated content is stored as values.

## **Example Output**
Given an image with various hormone names and gland details, the output dictionary could look like:

```json
{
    "Hypothalamus": "TRH, CRH, GHRH, Dopamine, Somatostatin, Vasopressin",
    "Pineal gland": "Melatonin",
    "Pituitary gland": "GH, TSH, ACTH, FSH, MSH, LH, Prolactin, Oxytocin, Vasopressin",
    "Thyroid and Parathyroid": "T3, T4, Calcitonin, PTH",
    "Thymus": "Thymopoietin",
    "Liver": "IGF, THPO",
    ...
}
```

## **Future Enhancements**
- **Deep Learning for Header-Content Classification**: Replace heuristic-based separation with a supervised model trained to classify headers and content based on labeled data.
- **Custom OCR Training**: Train a custom OCR model to handle domain-specific challenges, improving text extraction accuracy.
- **NLP-Based Content Structuring**: Use advanced NLP techniques (e.g., named entity recognition, text summarization) for better content grouping and organization.

## **Conclusion**
This solution is a hybrid of **AI/ML techniques** combined with traditional image processing. It leverages the power of OCR, combined with rules and heuristics, to extract and structure text effectively from images.

## **Contact**
For issues or enhancements, please open an issue in the repository or contact [Your Name] at [Your Email].

---

Feel free to modify it to include any specific instructions or details!
