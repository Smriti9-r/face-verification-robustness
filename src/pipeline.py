from detection import FaceDetector
from embedding import FaceEmbedder
from similarity import cosine_similarity


class FaceRecognitionPipeline:

    def __init__(self):
        self.detector = FaceDetector(min_face_size=20)
        self.embedder = FaceEmbedder()

    def get_embedding(self, image_path):
        boxes, probabilities = self.detector.detect(image_path)

        if boxes is None or len(boxes) == 0:
            return None

        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        x1, y1, x2, y2 = boxes[0]

        face = image.crop((
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        ))

        embedding = self.embedder.generate_embedding(face)

        return embedding

    def compare_faces(self, image_a, image_b):
        embedding_a = self.get_embedding(image_a)
        embedding_b = self.get_embedding(image_b)

        if embedding_a is None or embedding_b is None:
            return None

        score = cosine_similarity(
            embedding_a,
            embedding_b
        ).item()

        return score