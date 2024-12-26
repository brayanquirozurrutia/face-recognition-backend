import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from scipy.spatial.distance import cosine

# Init FaceNet
model = InceptionResnetV1(pretrained="vggface2").eval()

def extract_embedding(image: np.ndarray):
    """
    Extract embeddings from an image using FaceNet
    :param image: Image to process
    :return: Embeddings
    """

    # Preprocess the image (normalize, transpose, add batch dimension)
    image_tensor = (torch.tensor(np.transpose(image, (2, 0, 1)) / 255.0, dtype=torch.float32) - 0.5) * 2
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Extract embeddings
    with torch.no_grad():
        embedding = model(image_tensor).detach().numpy()

    return embedding.flatten()  # Flatten the embeddings to store in the database in 1D

def compare_embeddings(embedding1, embedding2, threshold=0.4):
    """
    Compare two embeddings using cosine similarity
    :param embedding1: The first embedding
    :param embedding2: The second embedding
    :param threshold: The threshold to consider the embeddings as the same person
    :return: True if the embeddings are from the same person, False otherwise
    """
    distance = cosine(embedding1, embedding2)
    return distance < threshold
