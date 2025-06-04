from flask import Flask, request, send_file, abort
from PIL import Image
import math
import io

app = Flask(__name__)

# Размер грани куба
CUBE_FACE_SIZE = 512

# Поля, которые присылает клиент
REQUIRED_FIELDS = ["front", "back", "left", "right", "top", "down"]

def load_cube_faces(files_dict):
    faces = {}
    for face in REQUIRED_FIELDS:
        if face not in files_dict:
            raise ValueError(f"Missing field: {face}")
        img = Image.open(files_dict[face]).convert("RGB")
        img = img.resize((CUBE_FACE_SIZE, CUBE_FACE_SIZE))
        faces[face] = img
    return faces

def sample_from_cube(faces_dict, dx, dy, dz):
    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
    if abs_dx >= abs_dy and abs_dx >= abs_dz:
        if dx > 0:
            face = "right"
            u = (-dz / abs_dx + 1) / 2
            v = (-dy / abs_dx + 1) / 2
        else:
            face = "left"
            u = (dz / abs_dx + 1) / 2
            v = (-dy / abs_dx + 1) / 2
    elif abs_dy >= abs_dx and abs_dy >= abs_dz:
        if dy > 0:
            face = "top"
            u = (dx / abs_dy + 1) / 2
            v = (dz / abs_dy + 1) / 2
        else:
            face = "down"
            u = (dx / abs_dy + 1) / 2
            v = (-dz / abs_dy + 1) / 2
    else:
        if dz > 0:
            face = "front"
            u = (dx / abs_dz + 1) / 2
            v = (-dy / abs_dz + 1) / 2
        else:
            face = "back"
            u = (-dx / abs_dz + 1) / 2
            v = (-dy / abs_dz + 1) / 2

    img = faces_dict[face]
    px = min(CUBE_FACE_SIZE - 1, max(0, int(u * (CUBE_FACE_SIZE - 1))))
    py = min(CUBE_FACE_SIZE - 1, max(0, int(v * (CUBE_FACE_SIZE - 1))))
    return img.getpixel((px, py))

def create_equirectangular(faces_dict, out_width=1024, out_height=512):
    equi = Image.new("RGB", (out_width, out_height))
    pixels = equi.load()

    for y in range(out_height):
        phi = math.pi * (0.5 - y / out_height)
        for x in range(out_width):
            theta = 2 * math.pi * (x / out_width - 0.5)
            dx = math.cos(phi) * math.sin(theta)
            dy = math.sin(phi)
            dz = math.cos(phi) * math.cos(theta)
            color = sample_from_cube(faces_dict, dx, dy, dz)
            pixels[x, y] = color

    return equi

@app.route("/stitch", methods=["POST"])
def stitch():
    try:
        faces = load_cube_faces(request.files)
    except Exception as e:
        return abort(400, f"Ошибка загрузки изображений: {e}")

    pano = create_equirectangular(faces, out_width=1024, out_height=512)
    buf = io.BytesIO()
    pano.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/jpeg",
        as_attachment=True,
        attachment_filename="panorama.jpg"
    )

@app.route("/", methods=["GET"])
def index():
    return """
    <h2>Cube2Sphere сервис запущен!</h2>
    <p>Чтобы получить панораму, сделайте POST /stitch с полями front, back, left, right, top, down.</p>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
