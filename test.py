import numpy as np

embedding_file = "facenet_embeddings1.npy"

try:
    data = np.load(embedding_file, allow_pickle=True).item()
    print("\n✅ Embeddings Loaded Successfully!\n")
    
    if data:
        for person, embedding in data.items():
            print(f"🔹 {person}: {embedding[:5]}... (Embedding size: {len(embedding)})")  # Print first 5 values
    else:
        print("❌ No embeddings found. Upload images and try again.")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")

