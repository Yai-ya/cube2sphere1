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
    1) Читает 6 PIL-изображений из request.files (ключи REQUIRED_FIELDS).
    2) Crop до квадрата (по меньшей стороне).
    3) Определяет минимальный face_size и ограничивает его MAX_FACE = 2048.
    4) Ресайзит каждую грань до (face_size × face_size).
    5) Конвертирует каждую грань в NumPy-массив dtype=uint8 (shape = face_size×face_size×3).
    6) Возвращает (faces_np, face_size), где faces_np[face] — NumPy-массив.
    """
    faces_np = {}
    face_size = None

    for face in REQUIRED_FIELDS:
        if face not in files_dict:
            raise ValueError(f"Missing field: {face}")

        # 1. Открываем PIL и приводим к RGB
        pil_img = Image.open(files_dict[face]).convert("RGB")
        w, h = pil_img.size

        # 2. Crop до квадрата
        if w != h:
            base = min(w, h)
            left = (w - base) // 2
            top = (h - base) // 2
            pil_img = pil_img.crop((left, top, left + base, top + base))
            w = h = base

        # 3. Итоговый face_size — минимум среди всех граней
        if face_size is None:
            face_size = w
        else:
            face_size = min(face_size, w)

        faces_np[face] = pil_img  # пока храним PIL, но потом заменим на NumPy

    # 4. Ограничиваем face_size до MAX_FACE
    MAX_FACE = 2048
    face_size = min(face_size, MAX_FACE)

    # 5. Ресайзим и конвертируем в NumPy
    for face in REQUIRED_FIELDS:
        pil_img = faces_np[face].resize((face_size, face_size))
        faces_np[face] = np.array(pil_img, dtype=np.uint8)
        # После конвертации PIL-объект будет освобождён (faces_np[face] — уже NumPy)

    return faces_np, face_size


def create_equirectangular_chunked(faces_np, face_size):
    """
    «Чанковый» рендер equirectangular панорамы (W=4*face_size, H=2*face_size),
    используя готовые NumPy-массивы faces_np[face] (shape = face_size×face_size×3).
    Разбивает по вертикали на полосы chunk_h и вставляет каждый кусок в итоговый PIL.Image.

    Возвращает PIL.Image размера (W, H).
    """
    H = 2 * face_size
    W = 4 * face_size

    # 1. Создаём пустой итоговый PIL-образ
    pano_img = Image.new("RGB", (W, H))

    # 2. Выбираем высоту чанка (256 строк — оптимальный баланс для face_size=2048)
    chunk_h = 256 if face_size >= 2048 else 128

    # 3. Сколько чанков по вертикали
    num_chunks = (H + chunk_h - 1) // chunk_h

    # 4. Предварительно строим θ-линейку один раз (W точек от –π до +π)
    xs = np.linspace(-math.pi, math.pi, W, endpoint=False, dtype=np.float32)

    # 5. Для каждого чанка (i=0..num_chunks-1) считаем y0, y1 и рендерим
    for i in range(num_chunks):
        y0 = i * chunk_h
        y1 = min(H, y0 + chunk_h)
        current_h = y1 - y0  # фактическая высота этого чанка

        # 5.1) Строим φ-линейку (φ от +π/2 до –π/2) для строк [y0..y1)
        start_phi = math.pi/2 - (y0 / H) * math.pi
        end_phi = math.pi/2 - ((y1 - 1) / H) * math.pi
        ys = np.linspace(start_phi, end_phi, current_h, dtype=np.float32)

        # 5.2) Двумерные массивы theta_chunk, phi_chunk (current_h × W)
        theta_chunk = np.repeat(xs[np.newaxis, :], current_h, axis=0)
        phi_chunk = np.repeat(ys[:, np.newaxis], W, axis=1)

        # 5.3) Вычисляем dx, dy, dz (каждый shape = current_h × W)
        cos_phi = np.cos(phi_chunk)
        dx = cos_phi * np.sin(theta_chunk)
        dy = np.sin(phi_chunk)
        dz = cos_phi * np.cos(theta_chunk)

        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)
        abs_dz = np.abs(dz)

        # 5.4) Подготавливаем карты map_x, map_y и индексы граней face_idx (current_h × W)
        map_x = np.zeros((current_h, W), dtype=np.float32)
        map_y = np.zeros((current_h, W), dtype=np.float32)
        face_idx = np.zeros((current_h, W), dtype=np.int8)

        # 5.5) Маски для того, какая грань куба “видна” в каждом пикселе
        mask_x = (abs_dx >= abs_dy) & (abs_dx >= abs_dz)
        right_mask = mask_x & (dx > 0)
        left_mask  = mask_x & (dx < 0)

        mask_y = (abs_dy >= abs_dx) & (abs_dy >= abs_dz)
        top_mask  = mask_y & (dy > 0)
        down_mask = mask_y & (dy < 0)

        mask_z = (abs_dz >= abs_dx) & (abs_dz >= abs_dy)
        front_mask = mask_z & (dz > 0)
        back_mask  = mask_z & (dz < 0)

        # 5.6) Для каждой маски считаем UV-координаты (u, v) и записываем в map_x, map_y, face_idx
        # — правая грань (индекс 0)
        if np.any(right_mask):
            u = (-dz[right_mask] / abs_dx[right_mask] + 1) * 0.5
            v = (-dy[right_mask] / abs_dx[right_mask] + 1) * 0.5
            map_x[right_mask] = u * (face_size - 1)
            map_y[right_mask] = v * (face_size - 1)
            face_idx[right_mask] = 0

        # — левая грань (1)
        if np.any(left_mask):
            u = ( dz[left_mask] / abs_dx[left_mask] + 1) * 0.5
            v = (-dy[left_mask] / abs_dx[left_mask] + 1) * 0.5
            map_x[left_mask] = u * (face_size - 1)
            map_y[left_mask] = v * (face_size - 1)
            face_idx[left_mask] = 1

        # — верхняя грань (2)
        if np.any(top_mask):
            u = (dx[top_mask] / abs_dy[top_mask] + 1) * 0.5
            v = (dz[top_mask] / abs_dy[top_mask] + 1) * 0.5
            map_x[top_mask] = u * (face_size - 1)
            map_y[top_mask] = v * (face_size - 1)
            face_idx[top_mask] = 2

        # — нижняя грань (3)
        if np.any(down_mask):
            u = (dx[down_mask] / abs_dy[down_mask] + 1) * 0.5
            v = (-dz[down_mask] / abs_dy[down_mask] + 1) * 0.5
            map_x[down_mask] = u * (face_size - 1)
            map_y[down_mask] = v * (face_size - 1)
            face_idx[down_mask] = 3

        # — передняя грань (4)
        if np.any(front_mask):
            u = (dx[front_mask] / abs_dz[front_mask] + 1) * 0.5
            v = (-dy[front_mask] / abs_dz[front_mask] + 1) * 0.5
            map_x[front_mask] = u * (face_size - 1)
            map_y[front_mask] = v * (face_size - 1)
            face_idx[front_mask] = 4

        # — задняя грань (5)
        if np.any(back_mask):
            u = (-dx[back_mask] / abs_dz[back_mask] + 1) * 0.5
            v = (-dy[back_mask] / abs_dz[back_mask] + 1) * 0.5
            map_x[back_mask] = u * (face_size - 1)
            map_y[back_mask] = v * (face_size - 1)
            face_idx[back_mask] = 5

        # 6) Для каждой из 6 граней выполняем remap и вставляем кусок в pano_img
        for idx, face_name in enumerate(["right", "left", "top", "down", "front", "back"]):
            mask = (face_idx == idx)
            if not np.any(mask):
                continue

            src_np = faces_np[face_name]  # NumPy-массив (face_size×face_size×3)
            mx = map_x.copy()
            my = map_y.copy()
            mx[~mask] = 0
            my[~mask] = 0

            # remap возвращает массив shape=(current_h, W, 3), dtype=uint8
            remapped_chunk = cv2.remap(
                src_np,
                mx,
                my,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # Преобразуем remapped_chunk → PIL и вставляем через маску
            chunk_img = Image.fromarray(remapped_chunk)
            # mask_img: grayscale-изображение (current_h × W), 255 там, где mask=True
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            pano_img.paste(chunk_img, (0, y0), mask_img)

        # После выхода из цикла i все массивы chunk (dx, dy, dz, map_x, map_y, face_idx, remapped_chunk)
        # освобождаются (сборщик мусора), а в памяти остаётся только pano_img и faces_np.

    return pano_img


@app.route("/stitch", methods=["POST"])
def stitch():
    try:
        faces_np, face_size = load_cube_faces(request.files)
    except Exception as e:
        return abort(400, f"Ошибка загрузки изображений: {e}")

    pano = create_equirectangular_chunked(faces_np, face_size)

    buf = io.BytesIO()
    pano.save(buf, format="JPEG", quality=95)  # или quality=90 для меньшего веса
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/jpeg",
        as_attachment=True,
        download_name="panorama.jpg"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
