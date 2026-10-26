from PIL import Image
from facenet_pytorch import MTCNN


class FaceDetector:

    def __init__(self, min_face_size=20):
        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=min_face_size
        )

    def detect(self, image_path):

        image = Image.open(image_path).convert("RGB")

        boxes, probabilities = self.mtcnn.detect(image)

        valid_boxes = []
        valid_probabilities = []

        for box, probability in zip(boxes, probabilities):

            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            if width >= 50 and height >= 50:
                valid_boxes.append(box)
                valid_probabilities.append(probability)

        return valid_boxes, valid_probabilities