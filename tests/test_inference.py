from src.model.inference import InferenceEngine

class MockYOLO:
    def predict(self, image, conf_thres=0.5):
        return [
            {"bbox": [0, 0, 100, 100], "confidence": 0.8, "class_id": 0},
            {"bbox": [10, 10, 50, 50], "confidence": 0.3, "class_id": 1}
        ]

def test_mocked_inference_threshold():
    engine = InferenceEngine(MockYOLO())
    results = engine.run("dummy", conf_thres=0.5)
    assert len(results) == 1
