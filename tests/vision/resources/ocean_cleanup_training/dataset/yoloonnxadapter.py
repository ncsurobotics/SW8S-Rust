import cv2
import numpy as np
import onnxruntime

class YoloONNXAdapter:
    def __init__(self, model_path="best.onnx", input_size=640, conf_thresh=0.4):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_thresh = conf_thresh

    def predict(self, image_path):
        # Load and preprocess
        image = cv2.imread(image_path)
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, (self.input_size, self.input_size))
        blob = resized.transpose(2, 0, 1)[None] / 255.0  # [1,3,640,640]
        blob = blob.astype(np.float32)

        # Run inference
        output = self.session.run(None, {self.input_name: blob})[0][0]

        # Parse results
        results = []
        for x, y, w, h, conf, cls_id in output:
            if conf < self.conf_thresh:
                continue
            # Rescale from [0-640] to original image size
            cx = x * orig_w / self.input_size
            cy = y * orig_h / self.input_size
            label = "shark" if int(cls_id) == 0 else "sawfish"
            results.append({
                "class_id": int(cls_id),
                "class_name": label,
                "confidence": round(float(conf), 3),
                "center": [round(cx, 1), round(cy, 1)]
            })

        return results

