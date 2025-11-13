# import all the libraries
import math
import numpy as np
import scipy
from PIL import Image
import torch
import torchvision.transforms as tforms
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
from diffusers.models import AutoencoderKL
import gradio as gr

# load SDXL pipeline
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained("mhdang/dpo-sdxl-text2image-v1", subfolder="unet", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, vae=vae, torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# watermarking helper functions. paraphrased from the reference impl of arXiv:2305.20030

def circle_mask(size=128, r=16, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0)**2 + (y-y0)**2)<= r**2

def get_pattern(shape, w_seed=999999):
    g = torch.Generator(device=pipe.device)
    g.manual_seed(w_seed)
    gt_init = pipe.prepare_latents(1, pipe.unet.in_channels,
                                   1024, 1024,
                                   pipe.unet.dtype, pipe.device, g)
    gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
    # ring pattern. paper found this to be effective
    gt_patch_tmp = gt_patch.clone().detach()
    for i in range(shape[-1] // 2, 0, -1):
        tmp_mask = circle_mask(gt_init.shape[-1], r=i)
        tmp_mask = torch.tensor(tmp_mask)
        for j in range(gt_patch.shape[1]):
            gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch

def transform_img(image):
    tform = tforms.Compose([tforms.Resize(1024),tforms.CenterCrop(1024),tforms.ToTensor()])
    image = tform(image)
    return 2.0 * image - 1.0

# hyperparameters
shape = (1, 4, 128, 128)
w_seed = 7433 # TREE :)
w_channel = 0
w_radius = 16 # the suggested r from section 4.4 of paper

# get w_key and w_mask
np_mask = circle_mask(shape[-1], r=w_radius)
torch_mask = torch.tensor(np_mask).to(pipe.device)
w_mask = torch.zeros(shape, dtype=torch.bool).to(pipe.device)
w_mask[:, w_channel] = torch_mask
w_key = get_pattern(shape, w_seed=w_seed).to(pipe.device)


def get_noise():
    # moved w_key and w_mask to globals

    # inject watermark
    init_latents = pipe.prepare_latents(1, pipe.unet.in_channels,
                                        1024, 1024,
                                        pipe.unet.dtype, pipe.device, None)
    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real
    # hot fix to prevent out of bounds values. will "properly" fix this later
    init_latents[init_latents == float("Inf")] = 4
    init_latents[init_latents == float("-Inf")] = -4

    return init_latents

def detect(image):
    # invert scheduler
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    # ddim inversion
    img = transform_img(image).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.13025
    inverted_latents = pipe(prompt="", latents=image_latents, guidance_scale=1, num_inference_steps=50, output_type="latent")
    inverted_latents = inverted_latents.images

    # calculate p-value instead of detection threshold. more rigorous, plus we can do a non-boolean output
    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))[w_mask].flatten()
    target = w_key[w_mask].flatten()
    inverted_latents_fft = torch.concatenate([inverted_latents_fft.real, inverted_latents_fft.imag])
    target = torch.concatenate([target.real, target.imag])

    sigma = inverted_latents_fft.std()
    lamda = (target ** 2 / sigma ** 2).sum().item()
    x = (((inverted_latents_fft - target) / sigma) ** 2).sum().item()
    p_value = scipy.stats.ncx2.cdf(x=x, df=len(target), nc=lamda)

    # revert scheduler
    pipe.scheduler = curr_scheduler

    if p_value == 0:
        return 1.0
    else:
        return max(0.0, 1-1/math.log(5/p_value,10))

def generate(prompt):
    return pipe(prompt=prompt, negative_prompt="monochrome", num_inference_steps=50, latents=get_noise()).images[0]

# optimize for speed
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
print(detect(generate("an astronaut riding a green horse"))) # warmup after jit

# actual gradio demo

def manager(input, progress=gr.Progress(track_tqdm=True)): # to prevent the queue from overloading
    if type(input) == str:
        return generate(input)
    elif type(input) == np.ndarray:
        image = Image.fromarray(input)
        percent = detect(image)
        return {"watermarked": percent, "not_watermarked": 1.0-percent}

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green",secondary_hue="green", font=gr.themes.GoogleFont("Fira Sans"))) as app:
    with gr.Row():
        gr.HTML('<center><p>Bad actors are using generative AI to destroy the livelihoods of real artists. We need transparency now.</p><h1><span style="font-size:1.5em">Introducing Dendrokronos 🌳</span></h1></center>')
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Generate\nType a prompt and hit Go. Dendrokronos will generate an invisibly-watermarked image.  \nYou can click the download button to save the finished image. Try it with the detector.")
            with gr.Group():
                with gr.Row():
                    gen_in = gr.Textbox(max_lines=1, placeholder='try "a majestic tree at sunset, oil painting"', show_label=False, scale=4)
                    gen_btn = gr.Button("Go", variant="primary", scale=0)
            gen_out = gr.Image(interactive=False, show_label=False)
            gen_btn.click(fn=manager, inputs=gen_in, outputs=gen_out)
        with gr.Column():
            gr.Markdown("# Detect\nUpload an image and hit Detect. Dendrokronos will predict the probability it was watermarked.  \nNote: Dendrokronos can only detect its own watermark. It won't detect other AIs, such as DALL-E.")
            det_out = gr.Label(show_label=False)
            with gr.Group():
                det_btn = gr.Button("Detect", variant="primary")
                det_in = gr.Image(interactive=True, sources=["upload","clipboard"], show_label=False)
            det_btn.click(fn=manager, inputs=det_in, outputs=det_out)
    with gr.Row():
        gr.HTML('<center><h1>&nbsp;</h1>Acknowledgements: Dendrokronos uses <a href="https://huggingface.co/mhdang/dpo-sdxl-text2image-v1">SDXL DPO 1.0</a> for the underlying image generation and <a href="https://arxiv.org/abs/2305.20030">an algorithm by UMD researchers</a> for the watermark technology.<br />Dendrokronos is a project by Devin Gulliver.</center>')

app.queue()
app.launch(show_api=False)
