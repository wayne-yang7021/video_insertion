from PIL import Image
from modules.models.segmentation_model import SAM2Config
from modules.detection.segmentation_sam2 import segment_image_sam2
from utils.get_video_first_frame import extract_first_frame

# video_image = extract_first_frame("data/videos/house_tour.mp4")
image = Image.open("data/pictures/living-room.jpg").convert("RGB")

res = segment_image_sam2(
    image=image,
    out_dir="./output",
    sam2_cfg=SAM2Config(
        checkpoint="checkpoints/sam2.1_hiera_small.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_s",  # 也接受 'configs/sam2.1/sam2.1_hiera_s.yaml'
        device=None,                         # 自動：cuda/mps/cpu
    ),
)

print("masks:", len(res.masks))
print("vis_path:", res.vis_path)
print("summary:", res.summary_json)
