import time
import numpy as np
import mlx.core as mx
from dataclasses import dataclass
from typing import Tuple
from tqdm import tqdm
from PIL import Image
from stable_diffusion import StableDiffusion
from deep_cache import DeepCacheSDHelper


@dataclass
class DeepcacheParams:
    deepcache = False
    cache_layer_id = 0
    cache_interval = 3
    cache_branch_id = 0
    skip_mode = "uniform"


def generate_image(
    sd: StableDiffusion,
    prompt: str,
    steps: int,
    cfg: float,
    deepcache_params: DeepcacheParams,
) -> Tuple[mx.array, float]:
    if deepcache_params.deepcache:
        helper = DeepCacheSDHelper(sd=sd)
        helper.cache_layer_id = deepcache_params.cache_layer_id
        helper.cache_interval = deepcache_params.cache_interval
        helper.cache_branch_id = deepcache_params.cache_branch_id
        helper.num_steps = steps
        helper.skip_mode = deepcache_params.skip_mode
        helper.wrap_modules()

    start_time = time.time()
    latents = sd.generate_latents(
        prompt,
        n_images=4,
        cfg_weight=cfg,
        num_steps=steps,
        seed=42,
    )
    for x_t in tqdm(latents, total=steps):
        mx.eval(x_t)

    # Decode them into images
    decoded = []
    for i in tqdm(range(0, 4, 1)):
        decoded.append(sd.decode(x_t[i : i + 1]))
        mx.eval(decoded[-1])

    # Arrange them on a grid
    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(1, B // 1, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(1 * H, B // 1 * W, C)
    x = (x * 255).astype(mx.uint8)

    t = time.time() - start_time

    return x, t


if __name__ == "__main__":
    prompt = "a photo of an F1 car on the circuit"
    steps = 50
    cfg = 7.5
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)

    params = DeepcacheParams()

    x, t = generate_image(sd, prompt, steps, cfg, params)

    # Save them to disc
    im = Image.fromarray(np.array(x))
    im.save("out.png")
    print("origin time: {:.2f}s".format(t))

    params.deepcache = True
    x, t = generate_image(sd, prompt, steps, cfg, params)

    # Save them to disc
    im = Image.fromarray(np.array(x))
    im.save("deepcache_out.png")
    print("deepcache time: {:.2f}s".format(t))
