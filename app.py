import math
import io
import numpy as np
import cv2               # pip install opencv-python
from PIL import Image
from flask import Flask, request, send_file, abort, render_template

app = Flask(__name__)

REQUIRED_FIELDS = ["front", "back", "left", "right", "top", "down"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def load_cube_faces(files_dict):
    """
    Загружает 6 изображений (front, back, left, right, top, down) из request.files.
    - Если картинка не квадратная, обрезает её (crop) до квадрата по меньшей стороне.
    - Вычисляет минимальный face_size среди всех граней и ограничивает его MAX_FACE=2048.
    - Ресайзит все грани до (face_size × face_size).
    Возвращает (faces_dict, face_size), где faces_dict[face] — PIL.Image размера face_size×face_size.
    """
    faces = {}
    face_size = None

    for face in REQUIRED_FIELDS:
        if face not in files_dict:
            raise ValueError(f"Missing field: {face}")

        img = Image.open(files_dict[face]).convert("RGB")
        w, h = img.size

        # Crop до квадрата (по меньшей стороне)
        if w != h:
            base = min(w, h)
            left = (w - base) // 2
            top = (h - base) // 2
            img = img.crop((left, top, left + base, top + base))
            w = h = base

        if face_size is None:
            face_size = w
        else:
            face_size = min(face_size, w)

        faces[face] = img

    # Ограничиваем максимальный размер грани (чтобы итог не стал слишком большим)
    MAX_FACE = 2048
    face_size = min(face_size, MAX_FACE)

    # Ресайзим все грани до face_size×face_size
    for face in REQUIRED_FIELDS:
        faces[face] = faces[face].resize((face_size, face_size))

    return faces, face_size


def create_equirectangular_chunked(faces_dict, face_size):
    """
    «Чанковый» рендер equirectangular панорамы (4*face_size × 2*face_size) по кусочкам,
    сохраняя каждый кусочек в итоговый PIL.Image вместо одного большого NumPy-буфера.

    Возвращает PIL.Image размера (4*face_size, 2*face_size).
    """
    H = 2 * face_size
    W = 4 * face_size

    # Создаём итоговый пустой PIL-образ
    pano_img = Image.new("RGB", (W, H))

    # Решаем, на какую высоту разбить чанки (256 или 512)
    # Чем меньше chunk_h — тем меньше памяти на каждый ремап, но чуть больше итераций.
    chunk_h = 256 if face_size >= 1024 else 128
    # Но если face_size=2048, рекомендовано 256
    if face_size >= 2048:
        chunk_h = 256

    # Количество чанков (округляем вверх)
    num_chunks = (H + chunk_h - 1) // chunk_h

    # Строим «θ-линейку» для всех W (от –π до +π)
    xs = np.linspace(-math.pi, math.pi, W, endpoint=False, dtype=np.float32)

    # Для каждой полосы y0..y1 генерируем «φ-линейку» и remap'им chunk
    for i in range(num_chunks):
        y0 = i * chunk_h
        y1 = min(H, y0 + chunk_h)
        current_h = y1 - y0  # фактическая высота этого чанка

        # 1) Строим φ-линейку (φ от +π/2 до –π/2) для строк [y0..y1)
        start_phi = math.pi/2 - (y0 / H) * math.pi
        end_phi = math.pi/2 - ((y1 - 1) / H) * math.pi
        ys = np.linspace(start_phi, end_phi, current_h, dtype=np.float32)

        # 2) Двумерные массивы theta_chunk, phi_chunk (current_h × W)
        theta_chunk = np.repeat(xs[np.newaxis, :], current_h, axis=0)
        phi_chunk = np.repeat(ys[:, np.newaxis], W, axis=1)

        # 3) dx, dy, dz
        cos_phi = np.cos(phi_chunk)
        dx = cos_phi * np.sin(theta_chunk)
        dy = np.sin(phi_chunk)
        dz = cos_phi * np.cos(theta_chunk)

        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)
        abs_dz = np.abs(dz)

        # 4) Инициализируем карты и индексы граней для чанка
        map_x = np.zeros((current_h, W), dtype=np.float32)
        map_y = np.zeros((current_h, W), dtype=np.float32)
        face_idx = np.zeros((current_h, W), dtype=np.int8)

        # 5) Маски по компоненте
        mask_x = (abs_dx >= abs_dy) & (abs_dx >= abs_dz)
        right_mask = mask_x & (dx > 0)
        left_mask  = mask_x & (dx < 0)

        mask_y = (abs_dy >= abs_dx) & (abs_dy >= abs_dz)
        top_mask  = mask_y & (dy > 0)
        down_mask = mask_y & (dy < 0)

        mask_z = (abs_dz >= abs_dx) & (abs_dz >= abs_dy)
        front_mask = mask_z & (dz > 0)
        back_mask  = mask_z & (dz < 0)

        # 6) По маскам заполняем u,v и face_idx
        # Правый (0)
        if np.any(right_mask):
            u = (-dz[right_mask] / abs_dx[right_mask] + 1) * 0.5
            v = (-dy[right_mask] / abs_dx[right_mask] + 1) * 0.5
            map_x[right_mask] = u * (face_size - 1)
            map_y[right_mask] = v * (face_size - 1)
            face_idx[right_mask] = 0

        # Левый (1)
        if np.any(left_mask):
            u = ( dz[left_mask] / abs_dx[left_mask] + 1) * 0.5
            v = (-dy[left_mask] / abs_dx[left_mask] + 1) * 0.5
            map_x[left_mask] = u * (face_size - 1)
            map_y[left_mask] = v * (face_size - 1)
            face_idx[left_mask] = 1

        # Верхний (2)
        if np.any(top_mask):
            u = (dx[top_mask] / abs_dy[top_mask] + 1) * 0.5
            v = (dz[top_mask] / abs_dy[top_mask] + 1) * 0.5
            map_x[top_mask] = u * (face_size - 1)
            map_y[top_mask] = v * (face_size - 1)
            face_idx[top_mask] = 2

        # Нижний (3)
        if np.any(down_mask):
            u = (dx[down_mask] / abs_dy[down_mask] + 1) * 0.5
            v = (-dz[down_mask] / abs_dy[down_mask] + 1) * 0.5
            map_x[down_mask] = u * (face_size - 1)
            map_y[down_mask] = v * (face_size - 1)
            face_idx[down_mask] = 3

        # Передний (4)
        if np.any(front_mask):
            u = (dx[front_mask] / abs_dz[front_mask] + 1) * 0.5
            v = (-dy[front_mask] / abs_dz[front_mask] + 1) * 0.5
            map_x[front_mask] = u * (face_size - 1)
            map_y[front_mask] = v * (face_size - 1)
            face_idx[front_mask] = 4

        # Задний (5)
        if np.any(back_mask):
            u = (-dx[back_mask] / abs_dz[back_mask] + 1) * 0.5
            v = (-dy[back_mask] / abs_dz[back_mask] + 1) * 0.5
            map_x[back_mask] = u * (face_size - 1)
            map_y[back_mask] = v * (face_size - 1)
            face_idx[back_mask] = 5

        # 7) С помощью cv2.remap получаем «кусок» remapped_chunk (current_h × W × 3)
        for idx, face_name in enumerate(["right", "left", "top", "down", "front", "back"]):
            mask = (face_idx == idx)
            if not np.any(mask):
                continue

            src_img = np.array(faces_dict[face_name])  # PIL.Image → NumPy (face_size×face_size×3)
            mx = map_x.copy()
            my = map_y.copy()
            mx[~mask] = 0
            my[~mask] = 0

            remapped_chunk = cv2.remap(
                src_img,
                mx,
                my,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )  # shape = (current_h, W, 3), uint8

            # 8) Вставляем этот remapped_chunk (только «mask»-пиксели) в итоговый PIL-образ
            # Преобразуем remapped_chunk → PIL.Image
            chunk_img = Image.fromarray(remapped_chunk)
            # Делаем «mask»-изображение (mode="L") для вставки
            # mask_img имеет размер current_h×W, где 255 там, где mask=True, и 0 везде
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            # Вставляем chunk_img в pano_img в позиции (0, y0) с использованием mask_img
            pano_img.paste(chunk_img, (0, y0), mask_img)

        # По завершении этого цикла все переменные chunk’а (dx, dy, dz, map_x, map_y, face_idx, remapped_chunk, chunk_img, mask_img)
        # выходят из области видимости, и память под них очищается.

    return pano_img


@app.route("/stitch", methods=["POST"])
def stitch():
    try:
        faces, face_size = load_cube_faces(request.files)
    except Exception as e:
        return abort(400, f"Ошибка загрузки изображений: {e}")

    # Чанковый рендер (fast и экономный по памяти)
    pano = create_equirectangular_chunked(faces, face_size)

    buf = io.BytesIO()
    pano.save(buf, format="PNG", optimize=True)  # Можно сменить на JPEG/TIFF, но PNG обычно OK
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name="panorama.png"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
