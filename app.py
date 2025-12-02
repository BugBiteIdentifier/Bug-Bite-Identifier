from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
from modelStructure import ResNet18#, PracticeCNN 

app = Flask(__name__)

# Load trained model (only)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=8).to(device)

#num_classes had to be specified because in the previous CNN the last layer was hardcoded to 8 outputs
#self.outputLayer = nn.Linear(128, 8), while with the ResNet model the final layer is configurable with
#a parameter

model.load_state_dict(torch.load("model/bugbite_cnn_model.pth", map_location=device))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

labels = ["Ant", "Bedbug", "Berry Bug", "Fleas", "Mosquito", "No", "Spider", "Tick"]

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
