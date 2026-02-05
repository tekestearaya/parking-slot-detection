from ultralytics import YOLO

class YOLOv8Model:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def predict(self, image, conf_thres=0.5):
        results = self.model(image, conf=conf_thres)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf),
                    "class_id": int(box.cls)
                })
        return detections
