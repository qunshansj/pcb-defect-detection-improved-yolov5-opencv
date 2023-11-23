python
class YOLOv5:
    def __init__(self, weights):
        self.weights = weights
        self.model = self._build_model()

    def _build_model(self):
        # build the YOLOv5 model using the provided weights
        # ...

    def detect(self, image):
        # perform object detection on the input image using the YOLOv5 model
        # ...

    def export(self, export_path):
        # export the YOLOv5 model to the specified export path
        # ...
