import cv2
from PIL import Image
from typing import List, Dict

def extract_first_frame(video_path: str) -> Image.Image:
    """
    從影片中提取第一幀，並轉為 PIL.Image 格式。
    """
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"❌ 無法從影片讀取第一幀：{video_path}")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    return image