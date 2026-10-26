import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms


class FaceEmbedder:

    def __init__(self):

        self.model = InceptionResnetV1(
            pretrained="vggface2"
        ).eval()

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def generate_embedding(self, face_image):

        face_image = face_image.convert("RGB")

        face_tensor = self.transform(face_image)

        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(face_tensor)

        return embedding