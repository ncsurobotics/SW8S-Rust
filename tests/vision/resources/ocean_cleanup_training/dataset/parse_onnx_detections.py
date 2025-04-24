from onnxruntime import InferenceSession
import numpy as np, cv2

session = InferenceSession("C:/Users/chris/Documents/learning/ncstate/underwater-robotics/SW8S-Rust/tests/vision/resources/ocean_cleanup_training/dataset/runs/detect/train/weights/best.onnx")
inp_name = session.get_inputs()[0].name

img = cv2.imread("C:/Users/chris/Documents/learning/ncstate/underwater-robotics/SW8S-Rust/tests/vision/resources/ocean_cleanup_training/dataset/images/train/sawfish_001_aug_0.png")
img = cv2.resize(img, (640, 640))
tensor = img.transpose(2, 0, 1)[None] / 255.0

output = session.run(None, {inp_name: tensor.astype(np.float32)})[0]
print(output.shape)  # (1, N, 6)
