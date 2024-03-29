import time
import numpy as np
import mlx.core as mx
from tqdm import tqdm
from PIL import Image
from stable_diffusion import StableDiffusion
from deep_cache import DeepCacheSDHelper


if __name__ == "__main__":
    prompt = "a photo of an F1 car on the circuit"
    steps = 50
    cfg = 7.5
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)

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

    origin_time = time.time() - start_time

    # Arrange them on a grid
    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(1, B // 1, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(1 * H, B // 1 * W, C)
    x = (x * 255).astype(mx.uint8)

    # Save them to disc
    im = Image.fromarray(np.array(x))
    im.save("out.png")
    print("origin time: {:.2f}s".format(origin_time))

    helper = DeepCacheSDHelper(sd=sd)
    start_time = time.time()
    helper.cache_layer_id = 0
    helper.cache_interval = 3
    helper.cache_branch_id =0
    helper.skip_mode = 'uniform'

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

    deepcache_time = time.time() - start_time

    # Arrange them on a grid
    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(1, B // 1, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(1 * H, B // 1 * W, C)
    x = (x * 255).astype(mx.uint8)

    # Save them to disc
    im = Image.fromarray(np.array(x))
    im.save("deepcache_out.png")
    print("deepcache time: {:.2f}s".format(deepcache_time))
