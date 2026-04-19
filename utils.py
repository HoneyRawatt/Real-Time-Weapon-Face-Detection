import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ArcFace (InsightFace buffalo_s / buffalo_l) always produces 512-d embeddings.
EXPECTED_EMBEDDING_DIM = 512


def load_known_faces(embeddings_file: str) -> dict:
    """
    Load face embeddings from a .npy dict file.

    Returns an empty dict (instead of raising) when the file is missing,
    corrupt, or contains embeddings from the old Facenet128/Facenet512 model
    so the rest of the system can keep running and the user gets a clear
    log message explaining what to do.
    """
    if not os.path.exists(embeddings_file):
        logger.warning(
            f"Embeddings file '{embeddings_file}' not found — "
            "starting with no enrolled faces.  "
            "Use the web UI (/create_encoding) to enrol people."
        )
        return {}

    try:
        data = np.load(embeddings_file, allow_pickle=True).item()
    except Exception as exc:
        logger.error(f"Failed to load '{embeddings_file}': {exc}")
        return {}

    if not isinstance(data, dict):
        logger.error("Embeddings file has unexpected format (expected dict) — ignoring.")
        return {}

    # Validate dimension for every stored embedding.
    # InsightFace buffalo_s ArcFace (w600k_mbf) produces 512-d vectors.
    # The old keras-facenet model produced 128-d vectors — reject those.
    # Note: InsightFace w600k_mbf outputs un-normalised vectors (norm≈20–30)
    # on some ONNX runtime versions, so we do NOT filter on L2-norm.
    # Cosine similarity (_cosine_sim) normalises both vectors explicitly,
    # so the comparison is correct regardless of output scale.
    for name, emb in data.items():
        dim = len(emb)

        if dim != EXPECTED_EMBEDDING_DIM:
            logger.warning(
                f"Dimension mismatch: '{name}' has {dim}-d embeddings but "
                f"ArcFace expects {EXPECTED_EMBEDDING_DIM}-d.  "
                "Please re-enrol all faces via /create_encoding."
            )
            return {}

    logger.info(
        f"Loaded {len(data)} enrolled face(s): {list(data.keys())}"
    )
    return data
