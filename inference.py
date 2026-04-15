import argparse
import torch
from torchvision import transforms
from PIL import Image
from model.alexnet import AlexNet

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

def predict(image_path, model_path="results/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = AlexNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)

    label = CLASSES[predicted.item()]
    score = confidence.item() * 100

    print(f"{label} ({score:.1f}%)")
    return label, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="이미지 경로")
    parser.add_argument("--model", default="results/best_model.pt", help="모델 가중치 경로")
    args = parser.parse_args()

    predict(args.image, args.model)
