import base64
import dataclasses
import io
import json
from dataclasses import dataclass

import diskcache
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

cache = diskcache.FanoutCache("/tmp/dgradui_fanout")


@dataclass
class PromptDefinition:
    prompt: str
    negative_prompt: str = ""
    seed: int = -1
    steps: int = 30
    sampler_index: str = "Euler a"
    width: int = 512
    height: int = 512
    restore_faces: bool = False


@cache.memoize()
def generate_sd_image(
    prompt_def: PromptDefinition,
):
    payload = {
        "prompt": prompt_def.prompt,
        "negative_prompt": prompt_def.negative_prompt,
        "seed": prompt_def.seed,
        "steps": prompt_def.steps,
        "sampler_index": prompt_def.sampler_index,
        "width": prompt_def.width,
        "height": prompt_def.height,
        "restore_faces": prompt_def.restore_faces,
    }

    s = requests.Session()

    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])

    s.mount("http://", HTTPAdapter(max_retries=retries))

    for i in range(5):
        try:
            resp = s.post(
                url="http://172.22.149.39:7860/sdapi/v1/txt2img",
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
        assert len(resp["images"]) == 1
        # This is annoying that the api returns the image in this format
        header = "data:image/png;base64,"
        for i in resp["images"]:
            decoded = base64.b64decode(i[len(header) :])
            img = Image.open(io.BytesIO(decoded))
            return img


def generate_frames(
    prompt_def: PromptDefinition, keyframes: list
) -> list[PromptDefinition]:
    frames = []

    for i, keyframe in enumerate(keyframes):
        if i == 0:
            continue

        prev_keyframe = keyframes[i - 1]

        delta_t = keyframe["t"]

        for t in range(delta_t):
            frame_values = {}
            for k, v in keyframe.items():
                if k == "t":
                    continue

                prev_v = prev_keyframe[k]

                if isinstance(v, str):
                    continue

                delta_v = v - prev_v

                new_v = prev_v + delta_v * (t / delta_t)

                frame_values[k] = new_v

            frame = dataclasses.replace(
                prompt_def, prompt=prompt_def.prompt.format(**frame_values)
            )

            frames.append(frame)

    return frames


def animate():
    prompt_def = PromptDefinition(
        prompt="portrait of danwinkler-2500 and (([grizzly::{grizzly}][:dog:{dog}] [bear::{bear}][:squirrel:{squirrel}][:chewbacca:{chewb}])), intricate, highly detailed, digital painting, artstation, concept art, matte, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha",
        negative_prompt="(((woman))), two people, scraggly beard, blurry, cropped, out of frame",
        seed=2173328752,
        width=768,
        height=512,
        restore_faces=True,
    )

    keyframes = [
        {"t": 0, "grizzly": 30, "bear": 30, "dog": 30, "squirrel": 30, "chewb": 30},
        {"t": 16, "grizzly": 16, "bear": 30, "dog": 30, "squirrel": 30, "chewb": 30},
        {"t": 16, "grizzly": 16, "bear": 16, "dog": 16, "squirrel": 30, "chewb": 30},
        {"t": 16, "grizzly": 16, "bear": 16, "dog": 30, "squirrel": 16, "chewb": 30},
        {"t": 16, "grizzly": 16, "bear": 16, "dog": 30, "squirrel": 16, "chewb": 16},
        {"t": 16, "grizzly": 30, "bear": 30, "dog": 30, "squirrel": 30, "chewb": 30},
    ]

    frame_defs = generate_frames(prompt_def, keyframes)

    frames = []
    for frame_def in tqdm(frame_defs):
        img = generate_sd_image(frame_def)
        frames.append(img)

    clip = ImageSequenceClip.ImageSequenceClip([np.asarray(i) for i in frames], fps=10)
    clip.write_videofile(
        "test.mp4",
        verbose=False,
        logger=None,
    )


if __name__ == "__main__":
    animate()
