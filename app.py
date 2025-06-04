from flask import Flask, request, send_file, abort, render_template
from PIL import Image
import math
import io

app = Flask(__name__)

# Размер каждой грани куба (будем ресайзить входные изображения до 512×512)
CUBE_FACE_SIZE = 512

# Имена полей, которые клиент обязан прислать в форме
REQUIRED_FIELDS = ["front", "back", "left", "right", "top", "down"]


@app.route("/", methods=["GET"])
def index():
    """
    Отдаём пользователю HTML-страницу с формой для загрузки 6-ти изображений.
    Шаблон лежит в папке templates/index.html
    """
    return render_template("index.html")


def load_cube_faces(files_dict):
    """
    Загружает из request.files шесть изображений по ключам REQUIRED_FIELDS.
    Возвращает словарь PIL.Image { "front": <Image>, "back": <Image>, ... }.
    Если какое-то поле не передано, бросает исключение ValueError.
    """
    faces = {}
    for face in REQUIRED_FIELDS:
        if face not in files_dict:
            raise ValueError(f"Missing field: {face}")
        img = Image.open(files_dict[face]).convert("RGB")
        img = img.resize((CUBE_FACE_SIZE, CUBE_FACE_SIZE))
        faces[face] = img
    return faces


def sample_from_cube(faces_dict, dx, dy, dz):
    """
    Для заданного нормализованного вектора (dx, dy, dz) определяет, какая грань куба
    (front/back/left/right/top/down) соответствует этому направлению, и берёт пиксель
    из соответствующей картинки.
    Возвращает RGB-кортеж (r, g, b).
    """
    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)

    # Определяем, какая из компонент имеет наибольшую абсолютную величину
    if abs_dx >= abs_dy and abs_dx >= abs_dz:
        # Смотрим на ±X: правая или левая грань
        if dx > 0:
            face = "right"
            u = (-dz / abs_dx + 1) / 2
            v = (-dy / abs_dx + 1) / 2
        else:
            face = "left"
            u = ( dz / abs_dx + 1) / 2
            v = (-dy / abs_dx + 1) / 2

    elif abs_dy >= abs_dx and abs_dy >= abs_dz:
        # Смотрим на ±Y: верхняя или нижняя грань
        if dy > 0:
            face = "top"
            u = (dx / abs_dy + 1) / 2
            v = (dz / abs_dy + 1) / 2
        else:
            face = "down"
            u = (dx / abs_dy + 1) / 2
            v = (-dz / abs_dy + 1) / 2

    else:
        # Смотрим на ±Z: передняя или задняя грань
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
    """
    Строит equirectangular-панораму (размером out_width×out_height) из шести граней куба.
    Возвращает PIL.Image размера (out_width, out_height).
    """
    equi = Image.new("RGB", (out_width, out_height))
    pixels = equi.load()

    for y in range(out_height):
        # phi: от +π/2 (y=0) до –π/2 (y=out_height)
        phi = math.pi * (0.5 - y / out_height)
        for x in range(out_width):
            # theta: от –π (x=0) до +π (x=out_width)
            theta = 2 * math.pi * (x / out_width - 0.5)

            dx = math.cos(phi) * math.sin(theta)
            dy = math.sin(phi)
            dz = math.cos(phi) * math.cos(theta)

            color = sample_from_cube(faces_dict, dx, dy, dz)
            pixels[x, y] = color

    return equi


@app.route("/stitch", methods=["POST"])
def stitch():
    """
    Принимает POST-запрос с шестью файлами (поля: front, back, left, right, top, down).
    Строит equirectangular-панораму и возвращает её как JPG.
    """
    try:
        faces = load_cube_faces(request.files)
    except Exception as e:
        return abort(400, f"Ошибка загрузки изображений: {e}")

    pano = create_equirectangular(faces, out_width=1024, out_height=512)

    # Записываем в байтовый буфер и отдаем как jpg
    buf = io.BytesIO()
    pano.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/jpeg",
        as_attachment=True,
        attachment_filename="panorama.jpg"
    )


if __name__ == "__main__":
    # Запуск локального сервера (для разработки)
    app.run(host="0.0.0.0", port=5000, debug=True)
