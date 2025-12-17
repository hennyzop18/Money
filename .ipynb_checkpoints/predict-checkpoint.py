import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from money_preprocess import crop_largest_object


def load_artifact(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    class_names = ckpt.get("class_names")
    image_size = int(ckpt.get("image_size", 224))
    return ckpt, class_names, image_size


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(model, pil_img: Image.Image, tfm, device):
    crop_res = crop_largest_object(pil_img)
    x = tfm(crop_res.image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return pred_idx, float(probs[pred_idx]), probs, crop_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/best_model.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    ckpt, class_names, image_size = load_artifact(args.model)

    if class_names is None:
        p = Path(args.model).parent / "class_names.json"
        if p.exists():
            class_names = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise RuntimeError("Missing class_names in checkpoint and artifacts/class_names.json")

    from torchvision import models
    import torch.nn as nn

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))
    model.load_state_dict(ckpt["model"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tfm = build_transform(image_size)
    pil_img = Image.open(args.image).convert("RGB")

    pred_idx, conf, probs, crop_res = predict_image(model, pil_img, tfm, device)

    pairs = list(enumerate(probs))
    pairs.sort(key=lambda x: x[1], reverse=True)

    print(f"used_crop={crop_res.used_crop} bbox={crop_res.bbox}")
    for i, p in pairs[: max(1, args.topk)]:
        print(f"{class_names[i]}\t{p:.4f}")

    print(f"pred={class_names[pred_idx]} conf={conf:.4f}")


if __name__ == "__main__":
    main()
