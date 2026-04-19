"""
CLI entry-point for SafeGuard real-time detection.

All detection logic lives in pipeline.py.  This file is intentionally thin
so that `python main.py` is the only thing needed to start the camera loop.
"""
from pipeline import detect_objects_in_realtime

if __name__ == "__main__":
    detect_objects_in_realtime()
