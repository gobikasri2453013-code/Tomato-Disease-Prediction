from flask import Flask, render_template, request
import joblib
import os
from src.feature_extraction import extract_features

app = Flask(__name__)

# ================= LOAD MODEL + CLASSES =================
data = joblib.load("model/tomato_model.pkl")
model = data["model"]
CLASSES = data["classes"]

print("Loaded classes:")
for i, cls in enumerate(CLASSES):
    print(i, cls)

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template(
                "index.html",
                result="No image uploaded",
                confidence=None
            )

        img_path = "temp.jpg"
        file.save(img_path)

        try:
            features = extract_features(img_path).reshape(1, -1)

            # LinearSVC does NOT support predict_proba
            idx = model.predict(features)[0]

            result = CLASSES[idx]
            confidence = "N/A"

        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
