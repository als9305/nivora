import base64
import os
import io
import uuid
from PIL import Image
import numpy as np

def base64_to_pil(base64_str: str) -> Image.Image:
    """Converts a base64 or Data URI string to a PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")

def base64_to_cv2(base64_str: str) -> np.ndarray:
    """Converts a base64 or Data URI string to an OpenCV BGR ndarray."""
    pil_img = base64_to_pil(base64_str)
    cv2_img = np.array(pil_img)
    # Convert RGB to BGR for OpenCV
    return cv2_img[:, :, ::-1].copy()

def cv2_to_base64(cv2_img: np.ndarray) -> str:
    """Converts an OpenCV BGR image to base64 JPEG data URI."""
    import cv2
    _, buffer = cv2.imencode('.jpg', cv2_img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def save_temp_file(data_bytes: bytes, extension: str = ".tmp") -> str:
    """Saves temporary byte data to disk and returns absolute path."""
    os.makedirs("scratch/temp", exist_ok=True)
    temp_path = os.path.join("scratch/temp", f"{uuid.uuid4().hex}{extension}")
    with open(temp_path, "wb") as f:
        f.write(data_bytes)
    return temp_path

def cleanup_temp_file(filepath: str):
    """Safely removes temporary files."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass
