import numpy as np

def load_known_faces(embeddings_file):
    with open(embeddings_file, "rb") as f:
        known_faces = np.load(f, allow_pickle=True).item()
    return known_faces
