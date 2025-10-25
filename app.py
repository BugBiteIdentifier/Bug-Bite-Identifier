from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
from modelStructure import PracticeCNN 

app = Flask(__name__)

# Load trained model (only)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PracticeCNN().to(device)
model.load_state_dict(torch.load("model/bugbite_cnn_model.pth", map_location=device))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

labels = ["Ant", "Bedbug", "Tick", "Berry Bug", "Spider", "Fleas", "No", "Mosquito"]

@app.route("/")
def home():
    return render_template("HomePage.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'photo' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['photo']
    img = Image.open(file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = labels[predicted.item()]

    return render_template("results.html", label=predicted_label)

if __name__ == "__main__":
    app.run(debug=True)
