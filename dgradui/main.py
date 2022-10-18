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
from requests.adapters import HTTPAdapter, Retry
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

from dgradui import model3d


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

    s = requests.Session()

    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])

    s.mount("http://", HTTPAdapter(max_retries=retries))

    for i in range(5):
        try:
            resp = s.post(
                url="http://172.22.149.39:7860/v1/txt2img",
                data=json.dumps(payload),
                timeout=10,
            ).json()
            break
        except requests.exceptions.ConnectionError as e:
            print("Connection error, retrying...")
    else:
        raise e

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
            print("Using cached image")
            image = cache["sd_image_cache"]
        else:
            print("Generating image...")
            image = generate_sd_image()
            print("saving image to cache")
            cache["sd_image_cache"] = image

        # Convert to numpy array
        image = np.array(image)

        # Crop to top left quadrant
        image = image[: image.shape[0] // 2, : image.shape[1] // 2]

        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold
        ret, thresh = cv2.threshold(image, threshold, 255, 0)

        # Invert?
        # thresh = 255 - thresh

        # Find Contours
        print("Finding contours...")
        contours, hierarchy = cv2.findContours(
            image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_KCOS
        )

        polygons = []
        # Convert contours to shapely polygons
        print("Fixing contours...")
        for contour in contours:
            contour = contour.squeeze()

            # Sometimes these contours are idiotic
            if len(contour) < 3:
                continue

            spoly = ShapelyPolygon(contour)
            spoly = spoly.simplify(1)

            # Just toss out invalid polygons
            if not spoly.is_valid:
                continue

            if spoly.area > 100:
                # Convert to list of lists
                polygon = np.array(spoly.exterior.coords).tolist()
                polygons.append(polygon)

        print("Generate solidpython definition")
        model = model3d.extrude_paths_into_base(polygons)

        # For testing
        # model3d.write_solid_to_file(model, "test.scad")
        # return image, None

        print("Generate STL")
        glb_path = model3d.solid_to_glb(model)

        return image, glb_path

    with gr.Row():
        with gr.Column():
            output_image = gr.Image()
        with gr.Column():
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
