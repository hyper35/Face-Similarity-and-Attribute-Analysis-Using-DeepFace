from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def face_detection(image_path):
    faces = DeepFace.extract_faces(image_path)
    if faces:
        return faces[0]['face']
    print(f"No faces detected in {image_path}")
    return None

def extract_embeddings(image_path, model_name="Facenet"):
    embedding = DeepFace.represent(image_path, model_name=model_name)[0]["embedding"]
    return np.array(embedding)

def calculate_cosine_distance(embedding1, embedding2):
    cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]
    cosine_dist = 1 - cosine_sim
    return cosine_dist

def calculate_linear_similarity_percentage(value):
    adjusted_value = 1 - value
    similarity_percentage = round(adjusted_value * 100, 2)
    return similarity_percentage

def calculate_exponential_similarity_percentage(value):
    adjusted_value = np.exp(-value)
    similarity_percentage = round(adjusted_value * 100, 2)
    return similarity_percentage

def plot_faces(img1_face, img2_face, img1_path, img2_path):
    axes = plt.subplots(1, 2, figsize=(10, 5))[1]
    axes[0].imshow(img1_face) if img1_face is not None else axes[0].text(0.5, 0.5, "No Face Detected", ha='center', va='center')
    axes[0].set_title(f"Image 1 ({os.path.basename(img1_path)})")
    axes[0].axis('off')
    
    axes[1].imshow(img2_face) if img2_face is not None else axes[1].text(0.5, 0.5, "No Face Detected", ha='center', va='center')
    axes[1].set_title(f"Image 2 ({os.path.basename(img2_path)})")
    axes[1].axis('off')
    
    plt.show()

def analyze_face(image_path):    
    return DeepFace.analyze(image_path, actions=['emotion', 'age', 'gender', 'race'])[0]

def print_analysis_results(analysis, image_label):
    print(f"\nAnalysis for {image_label}:")
    print(f"  Emotions: {analysis['dominant_emotion']}")
    print(f"  Age: {analysis['age']}")
    print(f"  Gender: {analysis['dominant_gender']}")
    print(f"  Race: {analysis['dominant_race']}")

def main(img1_path, img2_path, model_name="Facenet"):
    img1_face = face_detection(img1_path)
    img2_face = face_detection(img2_path)
    
    plot_faces(img1_face, img2_face, img1_path, img2_path)
    
    img1_embedding = extract_embeddings(img1_path, model_name)
    img2_embedding = extract_embeddings(img2_path, model_name)
    
    cosine_dist = calculate_cosine_distance(img1_embedding, img2_embedding)
    cosine_linear_percentage = calculate_linear_similarity_percentage(cosine_dist)
    cosine_exp_percentage = calculate_exponential_similarity_percentage(cosine_dist)
    
    result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name=model_name)
    deepface_dist = result['distance']
    deepface_linear_percentage = calculate_linear_similarity_percentage(deepface_dist)
    deepface_exp_percentage = calculate_exponential_similarity_percentage(deepface_dist)
    
    print(f"Custom Cosine Distance: {cosine_dist}")
    print(f"Custom Cosine Distance Linear Similarity Percentage: {cosine_linear_percentage}%")
    print(f"Custom Cosine Distance Exponential Similarity Percentage: {cosine_exp_percentage}%")
    print(f"\nDeepFace Distance: {deepface_dist}")
    print(f"DeepFace Linear Similarity Percentage: {deepface_linear_percentage}%")
    print(f"DeepFace Exponential Similarity Percentage: {deepface_exp_percentage}%\n")
    
    img1_analysis = analyze_face(img1_path)
    img2_analysis = analyze_face(img2_path)
    print_analysis_results(img1_analysis, "Image 1")
    print_analysis_results(img2_analysis, "Image 2")

if __name__ == "__main__":
    img1_path = "your path for img1"
    img2_path = "your path for img2"
    # models = ["Facenet", "VGG-Face", "DeepFace", "OpenFace", "Dlib", "ArcFace", "SFace"]
    model_name = "chose a model" 

    main(img1_path, img2_path, model_name)
