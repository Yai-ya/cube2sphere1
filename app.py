import os
import cv2
import numpy as np
import tempfile
from flask import Flask, request, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/stitch", methods=["POST"])
def stitch():
    expected_faces = ["front", "back", "left", "right", "top", "down"]
    face_paths = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        for face in expected_faces:
            if face not in request.files:
                return f"Missing face: {face}", 400
            file = request.files[face]
            filename = secure_filename(f"{face}.jpg")
            save_path = os.path.join(temp_dir, filename)
            file.save(save_path)
            face_paths[face] = save_path

        face_size = 2048
        width = 4 * face_size
        height = 2 * face_size
        result = np.zeros((height, width, 3), dtype=np.uint8)

        face_images = {face: cv2.imread(path) for face, path in face_paths.items()}

        def insert(face, x0, y0):
            img = cv2.resize(face_images[face], (face_size, face_size), interpolation=cv2.INTER_AREA)
            result[y0:y0+face_size, x0:x0+face_size] = img

        insert("top", face_size, 0)
        insert("left", 0, face_size)
        insert("front", face_size, face_size)
        insert("right", 2 * face_size, face_size)
        insert("back", 3 * face_size, face_size)
        insert("down", face_size, 2 * face_size)

        output_path = os.path.join(temp_dir, "panorama.jpg")
        cv2.imwrite(output_path, result)

        return send_file(
            output_path,
            mimetype="image/jpeg",
            as_attachment=True,
            download_name="panorama.jpg"
        )

if __name__ == "__main__":
    app.run(debug=True)
