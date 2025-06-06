# Fast equirectangular panorama stitching using pyvips
# Adapted from Stefan Keim’s “kubi” project (MIT License)
# Source: https://github.com/stefanix/kubi
# Adapted by: Yai, 2025

import pyvips

def stitch_cube_to_equirectangular_vips(face_paths: dict, out_path: str, face_size: int = 2048, width: int = None):
    """
    Склеивает панораму из 6 граней куба с помощью pyvips.
    face_paths: словарь с путями к изображениям: front, back, left, right, top, down
    out_path: путь к результирующему файлу
    face_size: размер одной стороны кубической проекции
    width: итоговая ширина эквидистантной проекции (если None — width = 4*face_size)

    Поддерживает форматы: .jpg, .png, .tif
    """
    if width is None:
        width = 4 * face_size
    height = 2 * face_size

    cube = {
        "front": pyvips.Image.new_from_file(face_paths["front"], access="sequential").resize(face_size / pyvips.Image.new_from_file(face_paths["front"]).width),
        "back": pyvips.Image.new_from_file(face_paths["back"], access="sequential").resize(face_size / pyvips.Image.new_from_file(face_paths["back"]).width),
        "left": pyvips.Image.new_from_file(face_paths["left"], access="sequential").resize(face_size / pyvips.Image.new_from_file(face_paths["left"]).width),
        "right": pyvips.Image.new_from_file(face_paths["right"], access="sequential").resize(face_size / pyvips.Image.new_from_file(face_paths["right"]).width),
        "top": pyvips.Image.new_from_file(face_paths["top"], access="sequential").resize(face_size / pyvips.Image.new_from_file(face_paths["top"]).width),
        "down": pyvips.Image.new_from_file(face_paths["down"], access="sequential").resize(face_size / pyvips.Image.new_from_file(face_paths["down"]).width),
    }

    # pyvips expects images in bands=3 RGB
    for k, img in cube.items():
        if img.bands == 4:
            cube[k] = img[:3]

    from kubi import Cubemap
    import numpy as np

    cubemap = Cubemap(cube["right"], cube["left"], cube["top"], cube["down"], cube["front"], cube["back"])
    eq = cubemap.to_equirectangular(width, height, "bilinear")
    eq.write_to_file(out_path)
