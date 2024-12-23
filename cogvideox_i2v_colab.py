# -*- coding: utf-8 -*-
"""CogVideoX-I2V-Colab.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX

## CogVideoX Image-to-Video

This notebook demonstrates how to run [CogVideoX-5b-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V) with 🧨 Diffusers on a free-tier Colab GPU.

Additional resources:
- [Docs](https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox)
- [Quantization with TorchAO](https://github.com/sayakpaul/diffusers-torchao/)
- [Quantization with Quanto](https://gist.github.com/a-r-r-o-w/31be62828b00a9292821b85c1017effa)

Note: If, for whatever reason, you randomly get an OOM error, give it a try on Kaggle T4 instances instead. I've found that Colab free-tier T4 can be unreliable at times. Sometimes, the notebook will run smoothly, but other times it will crash with an error 🤷🏻‍♂️

#### Install the necessary requirements
"""

!pip install diffusers transformers hf_transfer

# !pip install git+https://github.com/huggingface/accelerate
!pip install accelerate==0.33.0

"""#### Import required libraries

The following block is optional but if enabled, downloading models from the HF Hub will be much faster
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel

"""#### Load models and create pipeline

Note: `bfloat16`, which is the recommended dtype for running "CogVideoX-5b-I2V" will cause OOM errors due to lack of efficient support on Turing GPUs.

Therefore, we must use `float16`, which might result in poorer generation quality. The recommended solution is to use Ampere or above GPUs, which also support efficient quantization kernels from [TorchAO](https://github.com/pytorch/ao) :(
"""

model_id = "THUDM/CogVideoX-5b-I2V"

transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)

# Create pipeline and run inference
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
)

"""#### Enable memory optimizations

Note that sequential cpu offloading is necessary for being able to run the model on Turing or lower architectures. It aggressively maintains everything on the CPU and only moves the currently executing nn.Module to the GPU. This saves a lot of VRAM but adds a lot of overhead for inference, making generations extremely slow (1 hour+). Unfortunately, this is the only solution for running the model on Colab until efficient kernels are supported.
"""

pipe.enable_sequential_cpu_offload()
# pipe.vae.enable_tiling()

"""#### Generate!"""

prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")

video = pipe(image=image, prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]

export_to_video(video, "output.mp4", fps=8)