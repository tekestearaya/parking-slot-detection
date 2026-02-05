class InferenceEngine:
    def __init__(self, model):
        self.model = model

    def run(self, image, conf_thres=0.5):
        if image is None:
            return []

        outputs = self.model.predict(image, conf_thres)
        return [det for det in outputs if det["confidence"] >= conf_thres]
