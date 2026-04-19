"""
Face detection and recognition using InsightFace buffalo_s.

Detection:    SCRFD-500M  — handles faces up to ±90° rotation, partial occlusion,
                            variable lighting.  Replaces OpenCV Haar cascade.
Recognition:  ArcFace (MobileNet backbone, w600k_mbf)  — 512-d L2-normalised
                            embeddings.  Replaces keras-facenet FaceNet-128d.

The FaceAnalysis object is a module-level singleton initialised lazily on the
first call to detect_faces() or get_face_app().  InsightFace downloads the
buffalo_s model pack (~85 MB) to ~/.insightface/models/ on first use.
"""
import cv2
import numpy as np
import logging
import warnings

from config import FACE_THRESHOLD

# InsightFace internally uses a deprecated scikit-image API (SimilarityTransform.estimate).
# Suppress the FutureWarning so it doesn't pollute the console on every detected face.
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

logger = logging.getLogger(__name__)

_face_app = None


def get_face_app():
    """
    Return the InsightFace FaceAnalysis singleton.
    Initialises (and downloads model if needed) on the first call.
    """
    global _face_app
    if _face_app is not None:
        return _face_app

    try:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"],
        )
        # det_size=(320,320) is faster on CPU than the default (640,640)
        # while still catching faces down to ~20×20 px in a 640×480 frame.
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("InsightFace buffalo_s ready (SCRFD detector + ArcFace recogniser).")
    except Exception as exc:
        logger.error(f"Failed to initialise InsightFace: {exc}")
        raise
    return _face_app


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.  Returns 0.0 if either is zero."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def detect_faces(
    frame: np.ndarray,
    known_faces: dict,
    threshold: float = FACE_THRESHOLD,
) -> bool:
    """
    Detect all faces in *frame* and classify each against *known_faces*.

    Draws colour-coded bounding boxes + name/score labels on *frame* in-place:
      • Green  — recognised person (score ≥ threshold)
      • Red    — unknown person   (score < threshold or no enrolled faces)

    Args:
        frame:       BGR image (numpy ndarray) from OpenCV.
        known_faces: Dict {name: 512-d ArcFace embedding} from load_known_faces().
        threshold:   Minimum cosine similarity to accept a match (default 0.35).

    Returns:
        True if at least one unrecognised face is present, False otherwise.
    """
    app = get_face_app()
    faces = app.get(frame)

    if not faces:
        return False

    if not known_faces:
        logger.debug("No enrolled faces — all detected faces treated as Unknown.")

    unknown_detected = False

    for face in faces:
        embedding = face.embedding          # 512-d, L2-normalised by InsightFace
        x1, y1, x2, y2 = face.bbox.astype(int)

        best_name  = "Unknown"
        best_score = -1.0

        for name, stored_emb in known_faces.items():
            score = _cosine_sim(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_name  = name

        if best_score < threshold:
            best_name        = "Unknown"
            unknown_detected = True
            colour           = (0, 0, 255)    # red
        else:
            colour = (0, 255, 0)              # green

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{best_name} ({best_score:.2f})" if best_score >= 0 else "Unknown"
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2,
        )

    return unknown_detected
