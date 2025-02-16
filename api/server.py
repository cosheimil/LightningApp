from omegaconf import OmegaConf
import onnxruntime
import litserve as ls
from PIL import Image
import numpy as np
from typing import Dict, Any
from io import BytesIO
import base64


class SimpleLitAPI(ls.LitAPI):
    def setup(self, config):
        self.config = OmegaConf.load('config.yaml')
        self.ort_session = onnxruntime.InferenceSession(self.config.model.path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        return [img_array]

    def decode_request(self, request: Dict[str, Any]) -> Image.Image:
        image_bytes = BytesIO(base64.b64decode(request['image']))
        image = Image.open(image_bytes)
        return image

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        input_tensor = self.preprocess_image(image)
        ort_inputs = {self.input_name: input_tensor}
        ort_outs = self.ort_session.run(None, ort_inputs)
        probabilities = np.squeeze(ort_outs[0])
        print(probabilities.shape)
        classes = probabilities.argmax(axis=0)
        return {"predictions": classes.tolist()}

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return output


if __name__ == "__main__":
    server = ls.LitServer(
        SimpleLitAPI(),
        accelerator="auto",
        max_batch_size=1,
        timeout=False,
    )
    server.run(port=8000)