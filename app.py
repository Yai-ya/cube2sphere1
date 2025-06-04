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
    Загружает шесть квадратных изображений (front, back, left, right, top, down).
    Если картинка не квадратная — обрезает по центру до квадрата. 
    Затем вычисляет минимальный face_size из всех сторон, 
    но не больше MAX_FACE (2048 px). 
    После ресайзит все грани до (face_size × face_size).
    Возвращает (faces_dict, face_size).
    """
    faces = {}
    face_size = None

    # Читаем и обрезаем (crop) каждую грань до квадрата
    for face in REQUIRED_FIELDS:
        img = Image.open(files_dict[face]).convert("RGB")
        w, h = img.size

        # Crop до квадрата
        if w != h:
            base = min(w, h)
            left = (w - base) // 2
            top = (h - base) // 2
            img = img.crop((left, top, left + base, top + base))
            w = h = base

        # Фиксируем размер первой пакривали и находим минимальный размер среди всех
        if face_size is None:
            face_size = w
        else:
            face_size = min(face_size, w)

        faces[face] = img

    # Ограничиваем максимально допустимый размер грани
    MAX_FACE = 2048
    face_size = min(face_size, MAX_FACE)

    # Приводим все грани в faces к размеру (face_size × face_size)
    for face in REQUIRED_FIELDS:
        faces[face] = faces[face].resize((face_size, face_size))

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


import numpy as np
import cv2

def create_equirectangular_fast(faces_dict, face_size):
    """
    Быстрый рендер equirectangular-панорамы с помощью NumPy+OpenCV.
    faces_dict: {"front": PIL.Image, ...}, face_size: int (например, 2048).
    Возвращает PIL.Image размера (4*face_size × 2*face_size).
    """
    H = 2 * face_size
    W = 4 * face_size

    # 1) theta ∈ [−π, +π), phi ∈ [+π/2, −π/2]
    xs = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)     # θ
    ys = np.linspace(np.pi/2, -np.pi/2, H, endpoint=False, dtype=np.float32)  # φ

    theta = np.repeat(xs[np.newaxis, :], H, axis=0)   # shape (H, W)
    phi   = np.repeat(ys[:, np.newaxis], W, axis=1)   # shape (H, W)

    cos_phi = np.cos(phi)
    dx = cos_phi * np.sin(theta)
    dy = np.sin(phi)
    dz = cos_phi * np.cos(theta)

    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    abs_dz = np.abs(dz)

    # Карта перекладки и индекс грани
    map_x = np.zeros((H, W), dtype=np.float32)
    map_y = np.zeros((H, W), dtype=np.float32)
    face_idx = np.zeros((H, W), dtype=np.int8)

    # Маски для граней
    mask_x = (abs_dx >= abs_dy) & (abs_dx >= abs_dz)
    right_mask = mask_x & (dx > 0)
    left_mask  = mask_x & (dx < 0)

    mask_y = (abs_dy >= abs_dx) & (abs_dy >= abs_dz)
    top_mask  = mask_y & (dy > 0)
    down_mask = mask_y & (dy < 0)

    mask_z = (abs_dz >= abs_dx) & (abs_dz >= abs_dy)
    front_mask = mask_z & (dz > 0)
    back_mask  = mask_z & (dz < 0)

    # Правый (0)
    u = (-dz[right_mask] / abs_dx[right_mask] + 1) * 0.5
    v = (-dy[right_mask] / abs_dx[right_mask] + 1) * 0.5
    map_x[right_mask] = u * (face_size - 1)
    map_y[right_mask] = v * (face_size - 1)
    face_idx[right_mask] = 0

    # Левый (1)
    u = ( dz[left_mask] / abs_dx[left_mask] + 1) * 0.5
    v = (-dy[left_mask] / abs_dx[left_mask] + 1) * 0.5
    map_x[left_mask] = u * (face_size - 1)
    map_y[left_mask] = v * (face_size - 1)
    face_idx[left_mask] = 1

    # Верхний (2)
    u = (dx[top_mask] / abs_dy[top_mask] + 1) * 0.5
    v = (dz[top_mask] / abs_dy[top_mask] + 1) * 0.5
    map_x[top_mask] = u * (face_size - 1)
    map_y[top_mask] = v * (face_size - 1)
    face_idx[top_mask] = 2

    # Нижний (3)
    u = (dx[down_mask] / abs_dy[down_mask] + 1) * 0.5
    v = (-dz[down_mask] / abs_dy[down_mask] + 1) * 0.5
    map_x[down_mask] = u * (face_size - 1)
    map_y[down_mask] = v * (face_size - 1)
    face_idx[down_mask] = 3

    # Передний (4)
    u = (dx[front_mask] / abs_dz[front_mask] + 1) * 0.5
    v = (-dy[front_mask] / abs_dz[front_mask] + 1) * 0.5
    map_x[front_mask] = u * (face_size - 1)
    map_y[front_mask] = v * (face_size - 1)
    face_idx[front_mask] = 4

    # Задний (5)
    u = (-dx[back_mask] / abs_dz[back_mask] + 1) * 0.5
    v = (-dy[back_mask] / abs_dz[back_mask] + 1) * 0.5
    map_x[back_mask] = u * (face_size - 1)
    map_y[back_mask] = v * (face_size - 1)
    face_idx[back_mask] = 5

    # Создаём итоговый массив H×W×3
    pano = np.zeros((H, W, 3), dtype=np.uint8)

    # Проходим по каждой грани и применяем remap
    for i, face_name in enumerate(["right", "left", "top", "down", "front", "back"]):
        mask = (face_idx == i)
        if not np.any(mask):
            continue
        src_img = np.array(faces_dict[face_name])  # PIL→NumPy
        mx = map_x.copy()
        my = map_y.copy()
        mx[~mask] = 0
        my[~mask] = 0

        remapped = cv2.remap(
            src_img,
            mx,
            my,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        pano[mask] = remapped[mask]

    return Image.fromarray(pano)



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

    pano = create_equirectangular_fast(faces, face_size)

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
