import torch
from torchvision.transforms.functional import to_tensor
from torchmetrics.image import  StructuralSimilarityIndexMeasure
from PIL import Image

def calculate_ssim(image_path1, image_path2):
    # Load images
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')

    # Convert images to PyTorch tensors
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)

    # Calculate StructuralSimilarityIndexMeasure
    StructuralSimilarityIndexMeasure_value = StructuralSimilarityIndexMeasure()(img1_tensor, img2_tensor)

    return StructuralSimilarityIndexMeasure_value.item()

# Example usage
if "__name__" == "__main__":
    image_path1 = 'path/to/image1.jpg'
    image_path2 = 'path/to/image2.jpg'

    StructuralSimilarityIndexMeasure_score = calculate_ssim(image_path1, image_path2)
    print(f"StructuralSimilarityIndexMeasure Score: {StructuralSimilarityIndexMeasure_score}")
