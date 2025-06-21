import os
import torch
import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load numerical model
numerical_model = joblib.load("models/best_model.pkl")

# Class names for image classification
class_names = ["Normal", "Myocardial Infarction", "Arrhythmia", "Other"]

# Define the CNN model
class SimpleShuffleNet(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleShuffleNet, self).__init__()
        self.backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(self.backbone.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Load image model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model = SimpleShuffleNet(num_classes=4).to(device)
image_model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
image_model.eval()

# Image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Input from form
    age = float(request.form["age"])
    t_chol = float(request.form["t_chol"])
    hdl = float(request.form["hdl"])
    ldl = float(request.form["ldl"])
    tg = float(request.form["tg"])
    gender = int(request.form["gender"])

    # --- Compute risk score (same as training) ---
    is_senior = int(age >= 60)
    high_chol = int(t_chol >= 200)
    low_hdl = int(hdl < 40)
    high_ldl = int(ldl >= 130)
    high_tg = int(tg >= 150)
    risk_score = is_senior + high_chol + low_hdl + high_ldl + high_tg

    # Final input for numerical model
    numeric_input = np.array([[age, t_chol, hdl, ldl, tg, risk_score, gender]])
    numeric_prob = numerical_model.predict_proba(numeric_input)[0][1]

    # Handle image
    image_file = request.files["image"]
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
    image_file.save(image_path)

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = image_model(image_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        image_prob = np.max(probs)
        image_class = np.argmax(probs)
        image_label = class_names[image_class]

    # Late fusion decision
    final_prob = (numeric_prob + image_prob) / 2
    label = f"Likely: {image_label}" if final_prob >= 0.5 else "No significant heart disease"
    confidence = f"{final_prob * 100:.2f}%"

    return render_template("result.html", label=label, confidence=confidence, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
