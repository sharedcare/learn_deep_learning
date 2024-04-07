# DeepCache

[Original implementation](https://github.com/horseee/DeepCache/tree/master) uses Hugging Face's diffuser pipeline. This implementation applies DeepCache to the [Apple's Stable Diffusion repo](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion) with MLX framework.

## Get Started
`python3 main.py`

## Performance
### Stable Diffusion v2.1
1. Original (~120s)
   ![original](./out.png)
2. DeepCache (~60s)
   ![deepcache](./deepcache_out.png)
