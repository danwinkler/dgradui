import base64
import io
import json
import multiprocessing
import pathlib
import tempfile
import time
import traceback

import cv2
import diskcache
import gradio as gr
import numpy as np
import pycork
import requests
import trimesh
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm


def generate_sd_image(
    prompt="tesselations alhambra of a branch black and white closeup, intricate, highly detailed, 2D pattern",
    negative_prompt="blurry, scribble, sloppy, sketch, symmetry, symmetric, grayscale",
):
    payload = {
        "txt2imgreq": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            # optional parms
            "steps": 30,
            "sampler_name": "Euler a",
        }
    }

    resp = requests.post(
        url="http://172.22.149.39:7860/v1/txt2img", data=json.dumps(payload)
    ).json()

    if resp.get("images") is None:
        # uh oh, something is wrong. Show what it sent us, it's probably an error of some kind
        print("Error, got this returned:")
        print(resp)

    else:
        for i in resp["images"]:
            img = Image.open(io.BytesIO(base64.b64decode(i)))
            return img


def subtract(trimesh_a, trimesh_b):
    verts, faces = pycork.union(
        trimesh_a.vertices,
        trimesh_a.faces,
        trimesh_b.vertices,
        trimesh_b.faces,
    )

    return trimesh.Trimesh(vertices=verts, faces=faces)


def vectorize_app():
    cache = diskcache.Cache("/tmp/dgradui_cache")

    def execute(use_cache, threshold=127):
        if "sd_image_cache" in cache and use_cache:
            image = cache["sd_image_cache"]
        else:
            image = generate_sd_image()
            cache["sd_image_cache"] = image

        # Convert to numpy array
        image = np.array(image)

        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold
        ret, thresh = cv2.threshold(image, threshold, 255, 0)

        # Invert?
        # thresh = 255 - thresh

        # Find Contours
        contours, hierarchy = cv2.findContours(
            image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_KCOS
        )

        border = 10

        shape = trimesh.primitives.Box(
            extents=[image.shape[0] + border * 2, image.shape[1] + border * 2, 10],
            transform=trimesh.transformations.translation_matrix(
                [
                    (image.shape[0] + border * 2) / 2,
                    (image.shape[1] + border * 2) / 2,
                    0,
                ]
            ),
        )

        shape = None

        for i, contour in tqdm(enumerate(contours)):
            try:
                # Convert to shapely
                contour_poly = ShapelyPolygon(contour.squeeze())

                # contour_poly = contour_poly.simplify(5)

                print(len(contour), contour_poly.area, contour_poly.bounds)

                # Trimesh extrude
                mesh = trimesh.creation.extrude_polygon(contour_poly, height=20)

                # Translate
                tmat = trimesh.transformations.translation_matrix([border, border, -5])
                mesh.apply_transform(tmat)
                if shape == None:
                    shape = mesh
                else:
                    shape = subtract(shape, mesh)
            except Exception as e:
                traceback.print_exc()
                print("First fail")
            # else:
            #     break

        # Export to file
        try:
            temp_model_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
            shape.export(temp_model_file.name)
        except Exception as e:
            traceback.print_exc()

            return image, None

        return image, temp_model_file.name

    output_image = gr.Image()
    output_model = gr.Model3D()

    use_cache = gr.Checkbox(value=True, label="Use Cache")
    submit = gr.Button(label="Submit")
    submit.click(fn=execute, inputs=[use_cache], outputs=[output_image, output_model])


with gr.Blocks(title="DGradUI") as app:
    tabs = [
        ("Vectorize SD", vectorize_app),
    ]

    for title, func in tabs:
        with gr.Tab(label=title):
            func()


app.launch()
