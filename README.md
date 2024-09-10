# Face-Similarity-and-Attribute-Analysis-Using-DeepFace

This repository contains a Python script that uses the [DeepFace](https://github.com/serengil/deepface) library to perform face detection, embedding extraction, similarity comparison, and facial attribute analysis (emotion, age, gender, and race) between two images.

## Features

- **Face Detection**: Detects faces in the provided images using DeepFace.
- **Embedding Extraction**: Extracts facial embeddings using a pre-trained deep learning model (e.g., Facenet) to represent each detected face numerically.
- **Cosine Similarity Calculation**: Computes the cosine similarity between two facial embeddings to determine how similar the faces are.
- **DeepFace Verification**: Leverages DeepFace's built-in verification to compute the distance between faces and converts it to a similarity percentage.
- **Facial Analysis**: Analyzes the detected face for attributes like emotion, age, gender, and race.
- **Visualization**: Displays the detected faces side by side for visual comparison.

## How the Process Works

1. **Face Detection**:
   - The script starts by detecting faces in the two input images using DeepFace's `extract_faces` function. If a face is detected, it is returned for further analysis; otherwise, an appropriate message is displayed.

2. **Embedding Extraction**:
   - For each detected face, the script extracts facial embeddings using a deep learning model (e.g., Facenet) specified by the user. These embeddings are high-dimensional vectors that represent the facial features in a numerical format.

3. **Similarity Calculation**:
   - The script calculates the cosine similarity between the extracted embeddings of the two faces. Cosine similarity measures the angle between two vectors in a multi-dimensional space, which helps in understanding how similar or different the two faces are.
   - Two similarity scores are computed:
     - **Custom Cosine Distance**: A measure of distance between embeddings, converted to a linear similarity percentage.
     - **DeepFace Distance**: The default distance measure used by DeepFace, which is also converted to a linear similarity percentage.

4. **Facial Analysis**:
   - The script uses DeepFace's `analyze` function to determine various facial attributes like emotion, age, gender, and race for each detected face.

5. **Visualization**:
   - The detected faces are displayed side by side using Matplotlib, along with the analysis results printed in the console.

## Requirements

To use this script, you need to install the following Python libraries:

- `deepface`
- `matplotlib`
- `numpy`
- `scikit-learn`

You can install all the required packages by running:

```bash
pip install deepface matplotlib numpy scikit-learn
