import os
from src.model.inference import InferenceEngine
from src.model.yolo_wrapper import YOLOv8Model

DATASET_DIR = "data/processed"
WEIGHTS_PATH = "weights/best.pt"

def run_inference_on_dataset(conf_thres=0.5):
    model = YOLOv8Model(WEIGHTS_PATH)
    engine = InferenceEngine(model)

    results = {}

    for img_name in os.listdir(DATASET_DIR):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(DATASET_DIR, img_name)
            detections = engine.run(img_path, conf_thres)
            results[img_name] = detections

    return results

if __name__ == "__main__":
    outputs = run_inference_on_dataset()
    for img, dets in outputs.items():
        print(f"{img}: {len(dets)} detections")
