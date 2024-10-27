import argparse
import math
import os
import random
import threading
import time
import cv2
import tempfile
import imageio_ffmpeg
import gradio as gr
import torch
from PIL import Image
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
)
from diffusers.utils import load_video, load_image, export_to_video
from datetime import datetime, timedelta
from diffusers.image_processor import VaeImageProcessor
from openai import OpenAI
import moviepy.editor as mp
import utils
from rife_model import load_rife_model, rife_inference_with_latents
from huggingface_hub import hf_hub_download, snapshot_download
import gc
import torch._dynamo
from transformers import T5EncoderModel
from torchao.quantization import (
    autoquant,
    quantize_,
    int8_weight_only,
    int8_dynamic_activation_int8_weight,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    int4_weight_only,
    float8_dynamic_activation_float8_weight,
    float8_weight_only,
    fpx_weight_only,
)
from torchao.quantization.quant_api import PerRow
from torchao.sparsity import sparsify_

os.environ["TORCH_LOGS"] = "+dynamo,output_code,graph_breaks,recompiles"
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

# Set high precision for float32 matrix multiplications.
# This setting optimizes performance on NVIDIA GPUs with Ampere architecture (e.g., A100, RTX 30 series) or newer.
torch.set_float32_matmul_precision("high")

CONVERT_DTYPE = {
    "fp16": lambda module: module.to(dtype=torch.float16),
    "bf16": lambda module: module.to(dtype=torch.bfloat16),
    "fp8wo": lambda module: quantize_(module, float8_weight_only()),
    "fp8dq": lambda module: quantize_(module, float8_dynamic_activation_float8_weight()),
    "fp8dqrow": lambda module: quantize_(module, float8_dynamic_activation_float8_weight(granularity=PerRow())),
    "fp6_e3m2": lambda module: quantize_(module, fpx_weight_only(3, 2)),
    "fp5_e2m2": lambda module: quantize_(module, fpx_weight_only(2, 2)),
    "fp4_e2m1": lambda module: quantize_(module, fpx_weight_only(2, 1)),
    "int8wo": lambda module: quantize_(module, int8_weight_only()),
    "int8dq": lambda module: quantize_(module, int8_dynamic_activation_int8_weight()),
    "int4dq": lambda module: quantize_(module, int8_dynamic_activation_int4_weight()),
    "int4wo": lambda module: quantize_(module, int4_weight_only()),
    "autoquant": lambda module: autoquant(module, error_on_unseen=False),
    "sparsify": lambda module: sparsify_(module, int8_dynamic_activation_int8_semi_sparse_weight()),
}

device = "cuda" if torch.cuda.is_available() else "cpu"

hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir="model_real_esran")
snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")

# 解析命令行参数
parser = argparse.ArgumentParser(description="Generate a video from an image using CogVideoX")
parser.add_argument("--quantization_scheme", type=str, default="fp8", choices=list(CONVERT_DTYPE.keys()), help="The quantization scheme to use")
parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16', 'bfloat16')")
args = parser.parse_args()

quantization_scheme = args.quantization_scheme
dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

# 量化模型函数
def quantize_model(part, quantization_scheme):
    if quantization_scheme in CONVERT_DTYPE:
        return CONVERT_DTYPE[quantization_scheme](part)
    else:
        raise ValueError(f"Unsupported quantization scheme: {quantization_scheme}")

# 加载并量化模型组件
transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=dtype)
quantize_model(part=transformer, quantization_scheme=quantization_scheme)

# 加载并量化模型组件
text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="text_encoder", torch_dtype=dtype)
quantize_model(part=text_encoder, quantization_scheme=quantization_scheme)
transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="transformer", torch_dtype=dtype)
quantize_model(part=transformer, quantization_scheme=quantization_scheme)
vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=dtype)
quantize_model(part=vae, quantization_scheme=quantization_scheme)

# 初始化管道
#pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", text_encoder=text_encoder, transformer=transformer, vae=vae, torch_dtype=dtype)
#pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

pipe_image = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    transformer=transformer,
    vae=vae,
    #scheduler=pipe.scheduler,
    #tokenizer=pipe.tokenizer,
    text_encoder=text_encoder,
    torch_dtype=dtype
)

# 启用模型CPU卸载
pipe_image.enable_model_cpu_offload()

# 启用VAE切片和平铺
#pipe_image.vae.enable_slicing()
#pipe_image.vae.enable_tiling()


os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

upscale_model = utils.load_sd_upscale("model_real_esran/RealESRGAN_x4.pth", device)
frame_interpolation_model = load_rife_model("model_rife")

sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.
For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:
You will only ever output a single video description per user request.
When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.
Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = OpenAI()
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    "role": "assistant",
                    "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    "role": "assistant",
                    "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    "role": "assistant",
                    "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                },
                {
                    "role": "user",
                    "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"',
                },
            ],
            model="glm-4-plus",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=200,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt

def infer(
    prompt: str,
    image_input: str,
    seed: int = -1,
    progress=gr.Progress(track_tqdm=True),
):
    if seed == -1:
        seed = random.randint(0, 2**8 - 1)
    #print("image input in infer :")
    #print(image_input)
    image_input = Image.fromarray(image_input).resize(size=(720, 480))  # Convert to PIL
    #print("fa :")
    #print(image_input)
    image = load_image(image_input)
    #print("image :")
    #print(image)
    video_pt = pipe_image(
        image=image,
        prompt=prompt,
        num_inference_steps=50,
        num_videos_per_prompt=1,
        use_dynamic_cfg=True,
        output_type="pt",
        guidance_scale=7.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).frames
    #pipe_image.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return (video_pt, seed)

def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path

def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)

threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX-5B Huggingface Space🤗
           </div>
           <div style="text-align: center;">
               <a href="https://huggingface.co/THUDM/CogVideoX-5B">🤗 5B(T2V) Model Hub</a> |
               <a href="https://huggingface.co/THUDM/CogVideoX-5B-I2V">🤗 5B(I2V) Model Hub</a> |
               <a href="https://github.com/THUDM/CogVideo">🌐 Github</a> |
               <a href="https://arxiv.org/pdf/2408.06072">📜 arxiv </a>
           </div>
           <div style="text-align: center;display: flex;justify-content: center;align-items: center;margin-top: 1em;margin-bottom: .5em;">
              <span>If the Space is too busy, duplicate it to use privately</span>
              <a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space?duplicate=true"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-lg.svg" width="160" style="
                margin-left: .75em;
            "></a>
           </div>
           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experiential use only.
            </div>
           """)
    with gr.Row():
        with gr.Column():
            with gr.Accordion("I2V: Image Input (cannot be used simultaneously with video input)", open=False):
                image_input = gr.Image(label="Input Image (will be cropped to 720 * 480)")
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)

            with gr.Row():
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)", value=-1
                        )
                    with gr.Row():
                        enable_scale = gr.Checkbox(label="Super-Resolution (720 × 480 -> 2880 × 1920)", value=False)
                        enable_rife = gr.Checkbox(label="Frame Interpolation (8fps -> 16fps)", value=False)
                    gr.Markdown(
                        "✨In this demo, we use [RIFE](https://github.com/hzwer/ECCV2022-RIFE) for frame interpolation and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling(Super-Resolution).<br>&nbsp;&nbsp;&nbsp;&nbsp;The entire process is based on open-source solutions."
                    )

            generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)

    gr.Markdown("""
    <table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
        <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            🎥 Video Gallery
        </div>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, their delicate wings casting shadows on the petals below. In the background, a grand fountain cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth surface worn by the touch of countless visitors seeking a moment of tranquility in nature's embrace.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/cf5953ea-96d3-48fd-9907-c4708752c714" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A small boy, head bowed and determination etched on his face, sprints through the torrential downpour as lightning crackles and thunder rumbles in the distance. The relentless rain pounds the ground, creating a chaotic dance of water droplets that mirror the dramatic sky's anger. In the far background, the silhouette of a cozy home beckons, a faint beacon of safety and warmth amidst the fierce weather. The scene is one of perseverance and the unyielding spirit of a child braving the elements.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/fe0a78e6-b669-4800-8cf0-b5f9b5145b52" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/c182f606-8f8c-421d-b414-8487070fcfcb" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>An elderly gentleman, with a serene expression, sits at the water's edge, a steaming cup of tea by his side. He is engrossed in his artwork, brush in hand, as he renders an oil painting on a canvas that's propped up against a small, weathered table. The sea breeze whispers through his silver hair, gently billowing his loose-fitting white shirt, while the salty air adds an intangible element to his masterpiece in progress. The scene is one of tranquility and inspiration, with the artist's canvas capturing the vibrant hues of the setting sun reflecting off the tranquil sea.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/7db2bbce-194d-434d-a605-350254b6c298" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>In a dimly lit bar, purplish light bathes the face of a mature man, his eyes blinking thoughtfully as he ponders in close-up, the background artfully blurred to focus on his introspective expression, the ambiance of the bar a mere suggestion of shadows and soft lighting.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/62b01046-8cab-44cc-bd45-4d965bb615ec" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog's energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/d78e552a-4b3f-4b81-ac3f-3898079554f6" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>On a brilliant sunny day, the lakeshore is lined with an array of willow trees, their slender branches swaying gently in the soft breeze. The tranquil surface of the lake reflects the clear blue sky, while several elegant swans glide gracefully through the still water, leaving behind delicate ripples that disturb the mirror-like quality of the lake. The scene is one of serene beauty, with the willows' greenery providing a picturesque frame for the peaceful avian visitors.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/30894f12-c741-44a2-9e6e-ddcacc231e5b" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A Chinese mother, draped in a soft, pastel-colored robe, gently rocks back and forth in a cozy rocking chair positioned in the tranquil setting of a nursery. The dimly lit bedroom is adorned with whimsical mobiles dangling from the ceiling, casting shadows that dance on the walls. Her baby, swaddled in a delicate, patterned blanket, rests against her chest, the child's earlier cries now replaced by contented coos as the mother's soothing voice lulls the little one to sleep. The scent of lavender fills the air, adding to the serene atmosphere, while a warm, orange glow from a nearby nightlight illuminates the scene with a gentle hue, capturing a moment of tender love and comfort.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/926575ca-7150-435b-a0ff-4900a963297b" width="100%" controls autoplay loop></video>
            </td>
        </tr>
    </table>
        """)

    def generate(
        prompt,
        image_input,
        seed_value,
        scale_status,
        rife_status,
        progress=gr.Progress(track_tqdm=True)
    ):
        latents, seed = infer(
            prompt,
            image_input,
            seed=seed_value,
            progress=progress,
        )
        if scale_status:
            latents = utils.upscale_batch_and_concatenate(upscale_model, latents, device)
        if rife_status:
            latents = rife_inference_with_latents(frame_interpolation_model, latents)

        batch_size = latents.shape[0]
        batch_video_frames = []
        for batch_idx in range(batch_size):
            pt_image = latents[batch_idx]
            pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

            image_np = VaeImageProcessor.pt_to_numpy(pt_image)
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)
            batch_video_frames.append(image_pil)

        video_path = utils.save_video(batch_video_frames[0], fps=math.ceil((len(batch_video_frames[0]) - 1) / 6))
        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)
        seed_update = gr.update(visible=True, value=seed)

        return video_path, video_update, gif_update, seed_update

    def enhance_prompt_func(prompt):
        return convert_prompt(prompt, retry_times=1)

    generate_button.click(
        generate,
        inputs=[prompt, image_input, seed_param, enable_scale, enable_rife],
        outputs=[video_output, download_video_button, download_gif_button, seed_text],
    )

    enhance_button.click(enhance_prompt_func, inputs=[prompt], outputs=[prompt])

demo.queue(max_size=15)
demo.launch(share=True)

'''
https://github.com/THUDM/CogVideo/issues/245
torch>=2.5.0
torchao==0.5

don't use float16, should use bfloat16

python i2v_app_qm.py --quantization_scheme int8wo --dtype bfloat16
python i2v_app_qm.py --quantization_scheme fp8wo --dtype bfloat16
python i2v_app_qm.py --quantization_scheme bf16 --dtype bfloat16
python i2v_app_qm.py --quantization_scheme fp16 --dtype float16
'''
