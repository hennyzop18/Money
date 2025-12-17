import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from money_preprocess import crop_largest_object


@st.cache_resource
@st.cache_resource
def load_model(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    class_names = ckpt.get("class_names")
    image_size = int(ckpt.get("image_size", 224))

    if class_names is None:
        p = Path(model_path).parent / "class_names.json"
        class_names = json.loads(p.read_text(encoding="utf-8"))

    from torchvision import models
    import torch.nn as nn
    model = models.efficientnet_b0(weights=None) # weights=None vì chúng ta sẽ load từ checkpoint

    # 2. Lấy số features đầu vào từ lớp phân loại của EfficientNet
    in_features = model.classifier[1].in_features

    # 3. Thay thế lớp phân loại bằng một lớp mới
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    # --- KẾT THÚC THAY ĐỔI ---

    model.load_state_dict(ckpt["model"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, class_names, tfm, device

def predict(model, class_names, tfm, device, pil_img: Image.Image):
    crop_res = crop_largest_object(pil_img)
    x = tfm(crop_res.image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return pred_idx, float(probs[pred_idx]), probs, crop_res


def main():
    st.set_page_config(page_title="VN Money Detect", layout="centered")

    st.title("Nhận diện Tiền Việt Nam")

    model_path = st.sidebar.text_input("Model path", value="artifacts/best_model.pt")
    topk = st.sidebar.slider("Top-K", min_value=1, max_value=5, value=3)

    uploaded = st.file_uploader("Upload ảnh", type=["png", "jpg", "jpeg"]) 

    if not uploaded:
        st.info("Chọn ảnh để nhận diện.")
        return

    pil_img = Image.open(uploaded).convert("RGB")

    model, class_names, tfm, device = load_model(model_path)

    pred_idx, conf, probs, crop_res = predict(model, class_names, tfm, device, pil_img)

    st.subheader("Kết quả")
    st.write(f"Dự đoán: **{class_names[pred_idx]}** (confidence={conf:.3f})")

    if class_names[pred_idx] == "000000":
        st.warning("Không phát hiện tờ tiền (lớp 000000).")

    st.subheader("Ảnh")
    c1, c2 = st.columns(2)
    with c1:
        st.image(pil_img, caption="Ảnh gốc", use_container_width=True)
    with c2:
        st.image(crop_res.image, caption=f"Crop (used={crop_res.used_crop})", use_container_width=True)
        st.caption(f"bbox={crop_res.bbox}")

    pairs = list(enumerate(probs))
    pairs.sort(key=lambda x: x[1], reverse=True)

    st.subheader("Top dự đoán")
    for i, p in pairs[:topk]:
        st.write(f"{class_names[i]}: {p:.3f}")


if __name__ == "__main__":
    main()
