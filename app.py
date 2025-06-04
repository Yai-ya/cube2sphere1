from flask import Flask, request, send_file, abort, render_template
from PIL import Image
import math
import io

app = Flask(__name__)

# Список полей, которые клиент обязан прислать
REQUIRED_FIELDS = ["front", "back", "left", "right", "top", "down"]

@app.route("/", methods=["GET"])
def index():
    """
    Отдаёт пользователю HTML-страницу с формой для загрузки 6-ти изображений.
    Шаблон лежит в templates/index.html
    """
    return render_template("index.html")


def load_cube_faces(files_dict):
    """
    Читает из request.files шесть изображений по ключам REQUIRED_FIELDS.
    Проверяет, что они квадратные и одинакового размера:
      - Если картинка не квадратная, обрезаем её по центру до квадрата наименьшей стороны.
      - Если размеры у всех разные, приводим всё к размеру первой картинки (либо к минимальной общей стороне).
    Возвращает: (faces_dict, face_size) — словарь PIL.Image { "front": <Image>, ... }
        и face_size (целое) — базовый размер квадрата.
    """
    faces = {}
    face_size = None

    for face in REQUIRED_FIELDS:
        if face not in files_dict:
            raise ValueError(f"Missing field: {face}")

        # Открываем картинку и приводим к RGB
        img = Image.open(files_dict[face]).convert("RGB")
        w, h = img.size

        # Если не квадрат, обрезаем (crop) по центру до квадрата = min(w, h)
        if w != h:
            base = min(w, h)
            left = (w - base) // 2
            top = (h - base) // 2
            img = img.crop((left, top, left + base, top + base))
            w = h = base

        # Инициализируем face_size размерами первой картинки
        if face_size is None:
            face_size = w
        else:
            # Если текущая картинка отличается по размеру, ресайзим её до face_size×face_size
            if w != face_size:
                img = img.resize((face_size, face_size))
                w = h = face_size

        faces[face] = img

    return faces, face_size


def sample_from_cube(faces_dict, dx, dy, dz, face_size):
    """
    Для заданного нормализованного вектора (dx, dy, dz) определяет, к какой грани куба
    (front/back/left/right/top/down) этот вектор «смотрит». Затем вычисляет координаты
    (u, v) внутри этой грани и возвращает цвет пикселя из исходного изображения.
    """

    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)

    if abs_dx >= abs_dy and abs_dx >= abs_dz:
        # Смотрим на ±X: правая (right) или левая (left) грань
        if dx > 0:
            face = "right"
            u = (-dz / abs_dx + 1) / 2
            v = (-dy / abs_dx + 1) / 2
        else:
            face = "left"
            u = ( dz / abs_dx + 1) / 2
            v = (-dy / abs_dx + 1) / 2

    elif abs_dy >= abs_dx and abs_dy >= abs_dz:
        # Смотрим на ±Y: верхняя (top) или нижняя (down) грань
        if dy > 0:
            face = "top"
            u = (dx / abs_dy + 1) / 2
            v = (dz / abs_dy + 1) / 2
        else:
            face = "down"
            u = (dx / abs_dy + 1) / 2
            v = (-dz / abs_dy + 1) / 2

    else:
        # Смотрим на ±Z: передняя (front) или задняя (back) грань
        if dz > 0:
            face = "front"
            u = (dx / abs_dz + 1) / 2
            v = (-dy / abs_dz + 1) / 2
        else:
            face = "back"
            u = (-dx / abs_dz + 1) / 2
            v = (-dy / abs_dz + 1) / 2

    img = faces_dict[face]
    # Преобразуем (u,v) из [0,1] в координаты пикселей [0, face_size-1]
    px = min(face_size - 1, max(0, int(u * (face_size - 1))))
    py = min(face_size - 1, max(0, int(v * (face_size - 1))))
    return img.getpixel((px, py))


def create_equirectangular(faces_dict, face_size):
    """
    Собирает equirectangular-панораму из словаря faces_dict,
    используя размер face_size (квадратные грани face_size×face_size).
    Выходной размер: (4*face_size)×(2*face_size).
    """

    out_width = 4 * face_size
    out_height = 2 * face_size

    equi = Image.new("RGB", (out_width, out_height))
    pixels = equi.load()

    for y in range(out_height):
        # phi от +π/2 до –π/2
        phi = math.pi * (0.5 - y / out_height)
        for x in range(out_width):
            # theta от –π до +π
            theta = 2 * math.pi * (x / out_width - 0.5)

            dx = math.cos(phi) * math.sin(theta)
            dy = math.sin(phi)
            dz = math.cos(phi) * math.cos(theta)

            color = sample_from_cube(faces_dict, dx, dy, dz, face_size)
            pixels[x, y] = color

    return equi


@app.route("/stitch", methods=["POST"])
def stitch():
    """
    Ожидает POST /stitch с полями front, back, left, right, top, down.
    Читает эти шесть картинок, определяет face_size, собирает equirectangular и
    возвращает в виде JPG. Размер результата = (4×face_size)×(2×face_size).
    """
    try:
        faces, face_size = load_cube_faces(request.files)
    except Exception as e:
        return abort(400, f"Ошибка загрузки изображений: {e}")

    pano = create_equirectangular(faces, face_size)

    buf = io.BytesIO()
    pano.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/jpeg",
        as_attachment=True,
        download_name="panorama.jpg"
    )


if __name__ == "__main__":
    # Для локальной разработки; когда будет на Render, этот блок не используется.
    app.run(host="0.0.0.0", port=5000, debug=True)
