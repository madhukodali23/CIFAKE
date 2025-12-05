import os
import io
import json
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import numpy as np
import cv2

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================================================
# 1. MODEL ARCHITECTURE (SAME AS YOUR TRAINING CODE)
# =========================================================

class CNNBlock(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.input_shape = input_shape
        self.Layer = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=(3, 3)),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=(3, 3)),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, output_shape, kernel_size=(5, 5)),
            nn.BatchNorm2d(output_shape),
            nn.ReLU()
        )

    def get_output_shape(self, input_height, input_width):
        x = torch.randn(1, self.input_shape, input_height, input_width)
        return self.Layer(x).shape[2:]

    def forward(self, x):
        return self.Layer(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            stride=patch_size,
            kernel_size=patch_size,
            padding=0
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 att_dropout: float):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.MultiHeadAttention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=att_dropout,
            batch_first=True
        )

    def forward(self, x):
        x = self.LayerNorm(x)
        attn_output, _ = self.MultiHeadAttention(
            query=x,
            key=x,
            value=x,
            need_weights=False
        )
        return attn_output


class MultiLayerPreceptronBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 mlp_size: int,
                 dropout: float):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.MLP = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.LayerNorm(x)
        x = self.MLP(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 mlp_size: int,
                 attn_dropout: float,
                 mlp_dropout: float):
        super().__init__()
        self.MSA_Block = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            att_dropout=attn_dropout
        )
        self.MLP_Block = MultiLayerPreceptronBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=mlp_dropout
        )

    def forward(self, x):
        x = self.MSA_Block(x) + x
        x = self.MLP_Block(x) + x
        x = self.MSA_Block(x) + x
        return x


class ViTBlock(nn.Module):
    def __init__(self,
                 image_size: int,
                 in_channels: int,
                 patch_size: int,
                 num_transformer_layers: int,
                 embedding_dim: int,
                 mlp_size: int,
                 num_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 embedding_dropout: float,
                 num_classes: int = 2):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, embedding_dim),
            requires_grad=True
        )

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dim),
            requires_grad=True
        )

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.transformerencoder = nn.Sequential(*[
            TransformerEncoder(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout
            ) for _ in range(num_transformer_layers)
        ])

    def forward(self, x):
        batch_size = x.shape[0]

        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim=1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformerencoder(x)

        return x


class AttentionMechBlock(nn.Module):
    def __init__(self, dim, units=128):
        super().__init__()
        self.query = nn.Linear(dim, units)
        self.key = nn.Linear(dim, units)
        self.value = nn.Linear(dim, units)
        self.LayerNorm = nn.LayerNorm(normalized_shape=units)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax(Q @ K.transpose(1, 2) / (x.size(-1) ** 0.5), dim=-1)
        return self.LayerNorm((attn @ V).mean(dim=1))


class HybridModel(nn.Module):
    def __init__(self,
                 image_size: int,
                 in_channels: int,
                 hidden_units: int,
                 output_shape: int,
                 patch_size: int,
                 num_transformer_layers: int,
                 embedding_dim: int,
                 mlp_size: int,
                 num_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 embedding_dropout: float,
                 units: int = 128,
                 num_classes: int = 2):
        super().__init__()

        self.CNNBlock = CNNBlock(
            input_shape=3,
            hidden_units=hidden_units,
            output_shape=output_shape
        )

        self.cnn_output_height, self.cnn_output_width = self.CNNBlock.get_output_shape(
            image_size, image_size
        )

        self.ViTBlock = ViTBlock(
            image_size=self.cnn_output_height,
            in_channels=in_channels,
            patch_size=patch_size,
            num_transformer_layers=num_transformer_layers,
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            embedding_dropout=embedding_dropout,
            num_classes=num_classes
        )

        self.AttentionMechBlock = AttentionMechBlock(
            dim=embedding_dim,
            units=units
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=units, out_features=num_classes)
        )

    def forward(self, x):
        x = self.CNNBlock(x)
        x = self.ViTBlock(x)
        x = self.AttentionMechBlock(x)
        x = self.classifier(x)
        return x


# =========================================================
# 2. LOAD CONFIG, CLASS NAMES, MODEL WEIGHTS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "hybrid_model_config.json"), "r") as f:
    cfg = json.load(f)

with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    CLASS_NAMES = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridModel(**cfg).to(device)

state_dict = torch.load(
    os.path.join(BASE_DIR, "hybrid_model_weights.pth"),
    map_location=device
)
model.load_state_dict(state_dict)
model.eval()

# =========================================================
# 3. TRANSFORMS (SAME AS YOUR test_transform)
# =========================================================

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# 4. GRAD-CAM SETUP (CNN LAST CONV LAYER)
# =========================================================

# CNNBlock.Layer: [0]=Conv,1=BN,2=ReLU,3=Conv,4=BN,5=ReLU,6=Conv,7=BN,8=ReLU
target_layer = model.CNNBlock.Layer[6]  # last Conv2d

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)



def generate_gradcam(pil_image: Image.Image):
    """
    pil_image: original RGB image from user
    Returns: predicted index, confidence, probs dict, base64 grad-cam overlay
    """
    # For overlay, use resized image in [0,1]
    img_resized = pil_image.resize((32, 32))
    img_np = np.array(img_resized).astype(np.float32) / 255.0  # H,W,3 in [0,1]

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Forward for prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        label_name = CLASS_NAMES[pred_idx]

    # Grad-CAM
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_idx)]
    )[0]  # (H, W)

    

    # Overlay using pytorch-grad-cam helper
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)  # uint8

    # Convert to base64 PNG
    vis_pil = Image.fromarray(visualization)
    buf = io.BytesIO()
    vis_pil.save(buf, format="PNG")
    buf.seek(0)
    img_bytes = buf.read()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = "data:image/png;base64," + base64_str

    # Probabilities dict
    probs_dict = {
        CLASS_NAMES[i]: float(probs[i].item())
        for i in range(len(CLASS_NAMES))
    }

    return pred_idx, label_name, confidence, probs_dict, data_uri


# =========================================================
# 5. FLASK APP
# =========================================================

app = Flask(__name__)
CORS(app)  # allow React on localhost:3000


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    pred_idx, pred_label, confidence, probs_dict, gradcam_data_uri = generate_gradcam(
        image
    )

    return jsonify({
        "predicted_index": int(pred_idx),
        "predicted_class": pred_label,
        "confidence": float(confidence),
        "probabilities": probs_dict,
        "gradcam_image": gradcam_data_uri
    })


if __name__ == "__main__":
    print("✅ Backend running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
