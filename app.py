import math
import io
import numpy as np
import cv2              # pip install opencv-python
from PIL import Image
from flask import Flask, request, send_file, abort, render_template

app = Flask(__name__)

REQUIRED_FIELDS = ["front", "back", "left", "right", "top", "down"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def load_cube_faces(files_dict):
    """
    Загружает 6 изображений (ключи REQUIRED_FIELDS) и вырезает/ресайзит каждую до квадрата.
    Затем определяет минимальный face_size и ограничивает его MAX_FACE (2048).
    Возвращает: (faces_dict, face_size) где faces_dict[face] = PIL.Image (face_size×face_size).
    """
    faces = {}
    face_size = None

    for face in REQUIRED_FIELDS:
        if face not in files_dict:
            raise ValueError(f"Missing field: {face}")

        img = Image.open(files_dict[face]).convert("RGB")
        w, h = img.size

        # Crop до квадрата
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

    # Ограничиваем максимальный размер грани (чтобы итог не был слишком большим)
    MAX_FACE = 2048
    face_size = min(face_size, MAX_FACE)

    # Ресайзим все грани до face_size×face_size
    for face in REQUIRED_FIELDS:
        faces[face] = faces[face].resize((face_size, face_size))

    return faces, face_size


def create_equirectangular_fast(faces_dict, face_size):
    """
    Chunk-based рендер equirectangular с помощью NumPy+OpenCV.
    Разбивает итоговую панораму (H=2*face_size, W=4*face_size) на Н чанков по высоте,
    чтобы не держать в памяти все массивы целиком.
    Возвращает PIL.Image (RGB) размера (4*face_size × 2*face_size).
    """
    H = 2 * face_size
    W = 4 * face_size

    # Заранее создаём итоговый пустой массив NumPy, куда будем «рисовать» чанки
    pano = np.zeros((H, W, 3), dtype=np.uint8)

    # Выбираем высоту чанка (512 или 1024, чем меньше — тем меньше пиковая память, но чуть больше итераций)
    chunk_h = 512
    if face_size >= 2048:
        chunk_h = 512
    elif face_size >= 1024:
        chunk_h = 512
    else:
        chunk_h = 256

    # Вычисляем количество чанков по vertical (округляем вверх, если не делится ровно)
    num_chunks = (H + chunk_h - 1) // chunk_h

    # Предварительно подготовим «θ-линейку» и «φ-линейку» только один раз:
    xs = np.linspace(-math.pi, math.pi, W, endpoint=False, dtype=np.float32)  # θ
    # φ будем строить для каждой полосы отдельно

    for i in range(num_chunks):
        y0 = i * chunk_h
        y1 = min(H, y0 + chunk_h)
        current_h = y1 - y0  # фактическая высота этого чанка (возможно меньше, чем chunk_h)

        # 1) Строим φ-линейку для строк [y0..y1)
        #    φ = +π/2 − (y / H)*π, но для chunk-строк y = y0..y1−1:
        ys = np.linspace(math.pi/2 - (y0 / H) * math.pi,
                         math.pi/2 - ((y1 - 1) / H) * math.pi,
                         current_h,
                         dtype=np.float32)

        # 2) Строим двумерные массивы theta_chunk, phi_chunk размера (current_h, W)
        theta_chunk = np.repeat(xs[np.newaxis, :], current_h, axis=0)  # (current_h, W)
        phi_chunk = np.repeat(ys[:, np.newaxis], W, axis=1)            # (current_h, W)

        # 3) Вычисляем dx, dy, dz для CHUNK
        cos_phi = np.cos(phi_chunk)
        dx = cos_phi * np.sin(theta_chunk)  # (current_h, W)
        dy = np.sin(phi_chunk)
        dz = cos_phi * np.cos(theta_chunk)

        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)
        abs_dz = np.abs(dz)

        # 4) Заранее создаём «map_x_chunk», «map_y_chunk» и «face_idx_chunk» нужного размера
        map_x = np.zeros((current_h, W), dtype=np.float32)
        map_y = np.zeros((current_h, W), dtype=np.float32)
        face_idx = np.zeros((current_h, W), dtype=np.int8)

        # 5) Маски для граней на уровне CHUNK
        mask_x = (abs_dx >= abs_dy) & (abs_dx >= abs_dz)
        right_mask = mask_x & (dx > 0)
        left_mask  = mask_x & (dx < 0)

        mask_y = (abs_dy >= abs_dx) & (abs_dy >= abs_dz)
        top_mask  = mask_y & (dy > 0)
        down_mask = mask_y & (dy < 0)

        mask_z = (abs_dz >= abs_dx) & (abs_dz >= abs_dy)
        front_mask = mask_z & (dz > 0)
        back_mask  = mask_z & (dz < 0)

        # 6) Просчёт UV карт для каждой грани в chunk
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

        # 7) Для каждой грани делаем remap через OpenCV, только в пределах chunk
        for idx, face_name in enumerate(["right", "left", "top", "down", "front", "back"]):
            mask = (face_idx == idx)
            if not np.any(mask):
                continue
            src_img = np.array(faces_dict[face_name])  # PIL→NumPy (face_size×face_size×3)
            mx = map_x.copy()
            my = map_y.copy()
            # Чтобы remap не «таскал» пиксели вне маски, «обнулим» координаты вне
            mx[~mask] = 0
            my[~mask] = 0

            # remap для chunk: src_img → весь (current_h × W) массив (float32 карты)
            remapped = cv2.remap(
                src_img,
                mx,
                my,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )  # возвращает массив (current_h, W, 3)

            # Копируем пиксели remapped[mask] в итоговый pano[y0:y1, x0:x1]
            pano[y0:y1, :][mask] = remapped[mask]

        # После цикла „mask“ и remap локальные массивы (dx, dy, dz, map_x, map_y, face_idx, remapped)
        # выходят из области видимости, и Python освобождает их (garbage collector).

    # 8) В конце собираем PIL.Image из pano
    return Image.fromarray(pano)


@app.route("/stitch", methods=["POST"])
def stitch():
    try:
        faces, face_size = load_cube_faces(request.files)
    except Exception as e:
        return abort(400, f"Ошибка загрузки изображений: {e}")

    # Здесь вызываем “быстрый” чанковый рендер
    pano = create_equirectangular_fast(faces, face_size)

    buf = io.BytesIO()
    # Сохраняем в PNG без потерь; при желании можно использовать JPEG/ TIFF
    pano.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name="panorama.png"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
