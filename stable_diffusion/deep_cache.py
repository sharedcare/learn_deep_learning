import mlx.core as mx
from stable_diffusion import StableDiffusion


class DeepCacheSDHelper(object):
    def __init__(self, sd: StableDiffusion) -> None:
        self.sd = sd
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None
        self.num_steps = 0
        self.cache_interval = 1
        self.cache_layer_id = 0
        self.cache_block_id = 0
        self.skip_mode = "uniform"

    def is_skip_step(self, block_i, layer_i, blocktype="down"):
        self.start_timestep = (
            self.cur_timestep if self.start_timestep is None else self.start_timestep
        )  # For some pipeline that the first timestep != 0
        if self.skip_mode == "uniform":
            if (self.cur_timestep - self.start_timestep) % self.cache_interval == 0:
                return False
        if block_i > self.cache_block_id or blocktype == "mid":
            return True
        if block_i < self.cache_block_id:
            return False
        return (
            layer_i >= self.cache_layer_id
            if blocktype == "down"
            else layer_i > self.cache_layer_id
        )

    def wrap_unet_forward(self):
        self.function_dict["unet_forward"] = self.sd.unet.__call__
        def wrapped_forward(*args, **kwargs):
            self.cur_timestep = [t_prev for t, t_prev in self.sd.sampler.timesteps(self.num_steps, self.start_timestep)].index(
                args[1].item()
            )
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
            downsampler = getattr(block, "downsample", [])
            if downsampler:
                self.wrap_block_forward(
                    downsampler,
                    "downsample",
                    block_i,
                    len(getattr(block, "resnets", [])),
                )
            self.wrap_block_forward(block, "block", block_i, 0, blocktype="down")
        # 3. wrap midblock forward
        for block_i, block in enumerate(self.sd.unet.mid_blocks):
            attention = getattr(block, "attentions", [])
            if attention:
                self.wrap_block_forward(
                    attention, "mid_attentions", block_i, layer_i, blocktype="mid"
                )
            resnet = getattr(block, "resnets", [])
            if resnet:
                self.wrap_block_forward(
                    resnet, "mid_resnet", block_i, layer_i, blocktype="mid"
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
            upsampler = getattr(block, "upsample", [])
            if upsampler:
                self.wrap_block_forward(
                    upsampler, "upsample", block_num - block_i - 1, 0, blocktype="up"
                )
            self.wrap_block_forward(
                block, "block", block_num - block_i - 1, 0, blocktype="up"
            )
