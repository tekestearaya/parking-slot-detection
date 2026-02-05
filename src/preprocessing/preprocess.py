from src.utils.helpers import resize_image, normalize_image

def preprocess_image(image):
    image = resize_image(image)
    image = normalize_image(image)
    return image
