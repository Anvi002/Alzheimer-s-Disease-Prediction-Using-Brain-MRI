
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = {
    0: "Normal",
    1: "Alzheimers"
}

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image_tensor = eval_transform(image)
    return image_tensor.unsqueeze(0)
