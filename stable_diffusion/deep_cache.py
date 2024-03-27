import mlx.core as mx
from stable_diffusion.stable_diffusion import StableDiffusion


class DeepCacheSDHelper(object):
    def __init__(self, sd: StableDiffusion) -> None:
        self.sd = sd
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None

    def wrap_unet_forward(self):
        self.function_dict["unet_forward"] = self.sd.unet.__call__
        def wrapped_forward(*args, **kwargs):
            self.cur_timestep = list(self.sd.sampler.timesteps).index(args[1].item())
            result = self.function_dict['unet_forward'](*args, **kwargs)
            return result
        self.sd.unet.__call__ = wrapped_forward

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        self.function_dict[(blocktype, block_name, block_i, layer_i)] = block.__call__

        def wrapped_forward(*args, **kwargs):
            skip = self.is_skip_step(block_i, layer_i, blocktype)
            result = (
                self.cached_output[(blocktype, block_name, block_i, layer_i)]
                if skip
                else self.function_dict[(blocktype, block_name, block_i, layer_i)](
                    *args, **kwargs
                )
            )
            if not skip:
                self.cached_output[(blocktype, block_name, block_i, layer_i)] = result
            return result

        block.__call__ = wrapped_forward

    def wrap_modules(self):
        # 1. wrap unet forward
        self.wrap_unet_forward()
        # 2. wrap downblock forward
        for block_i, block in enumerate(self.sd.unet.down_blocks):
            for layer_i, attention in enumerate(getattr(block, "attentions", [])):
                self.wrap_block_forward(attention, "attentions", block_i, layer_i)
            for layer_i, resnet in enumerate(getattr(block, "resnets", [])):
                self.wrap_block_forward(resnet, "resnet", block_i, layer_i)
            for downsampler in (
                getattr(block, "downsample", []) if block.downsample else []
            ):
                self.wrap_block_forward(
                    downsampler,
                    "downsample",
                    block_i,
                    len(getattr(block, "resnets", [])),
                )
            self.wrap_block_forward(block, "block", block_i, 0, blocktype="down")
        # 3. wrap midblock forward
        self.wrap_block_forward(
            self.sd.unet.mid_block, "mid_block", 0, 0, blocktype="mid"
        )
        # 4. wrap upblock forward
        block_num = len(self.sd.unet.up_blocks)
        for block_i, block in enumerate(self.sd.unet.up_blocks):
            layer_num = len(getattr(block, "resnets", []))
            for layer_i, attention in enumerate(getattr(block, "attentions", [])):
                self.wrap_block_forward(
                    attention,
                    "attentions",
                    block_num - block_i - 1,
                    layer_num - layer_i - 1,
                    blocktype="up",
                )
            for layer_i, resnet in enumerate(getattr(block, "resnets", [])):
                self.wrap_block_forward(
                    resnet,
                    "resnet",
                    block_num - block_i - 1,
                    layer_num - layer_i - 1,
                    blocktype="up",
                )
            for upsampler in (
                getattr(block, "upsample", []) if block.upsample else []
            ):
                self.wrap_block_forward(
                    upsampler, "upsample", block_num - block_i - 1, 0, blocktype="up"
                )
            self.wrap_block_forward(
                block, "block", block_num - block_i - 1, 0, blocktype="up"
            )
