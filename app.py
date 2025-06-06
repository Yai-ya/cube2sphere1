# app.py — версия с pyvips
# Автор: Yai + адаптация OpenAI
# Склейка панорамы через pyvips + kubi

import os
import tempfile
from flask import Flask, request, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from equirectangular_vips import stitch_cube_to_equirectangular_vips

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
        # Сохраняем 6 изображений во временную папку
        for face in expected_faces:
            if face not in request.files:
                return f"Missing face: {face}", 400
            file = request.files[face]
            filename = secure_filename(f"{face}.jpg")
            save_path = os.path.join(temp_dir, filename)
            file.save(save_path)
            face_paths[face] = save_path

        # Выходной путь
        output_path = os.path.join(temp_dir, "panorama.jpg")

        # Склейка с помощью pyvips
        try:
            stitch_cube_to_equirectangular_vips(face_paths, output_path)
        except Exception as e:
            return f"Error during stitching: {str(e)}", 500

        # Отправляем результат
        return send_file(output_path, mimetype="image/jpeg", as_attachment=True, download_name="panorama.jpg")

if __name__ == "__main__":
    app.run(debug=True)
