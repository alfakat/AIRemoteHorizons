import cv2
import torch
import numpy as np
import gradio as gr
import db_examples
from briarmbg import BriaRMBG
from PIL import Image
from diffusers import StableDiffusionXLPipeline

"""
basic for StableDiffusion generation
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)

negative_prompt = 'Deformed, mutated, extra limbs, disfigured, ugly, bad anatomy, missing limbs,' \
                  'bad, immature, cartoon, anime, painting, mutant, body horror,'  \
                                                '(six fingers), (extra fingers), (bad hands),' \
                                                '(poorly drawn hands), (fused fingers), (too many fingers),' \
                                                '(unnatural hands), (disfigured hands)'
"""


device = "cuda" if torch.cuda.is_available() else "cpu"

""" This part cares about background splitting"""
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
rmbg.to('cuda')


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha



def generate_image(input_image, background, prompt, image_width, image_height, seed):

    color_photo = 'a color frontal photo of human'
    reccomended_prompt = '50mm portrait photography, camera lens in front, daylighting photography --beta --ar 3:2 --upbeta'
    input_fg, matting = run_rmbg(input_image)

    new_foreground_resized = cv2.resize(input_fg, (background.shape[1], background.shape[0]))
    result = cv2.addWeighted(background, 0.5, new_foreground_resized, 0.5, 0)

    """Once we found better solution to generate background, open it"""
    # image = pipe(prompt=color_photo + prompt + reccomended_prompt
    #              negative_prompt=negative_prompt,
    #              num_inference_steps=1,
    #              generator=torch.Generator(device=device).manual_seed(seed)).images[0]
    return result



my_theme = gr.Theme.from_hub("JohnSmith9982/small_and_pretty")
block = gr.Blocks(theme=my_theme).queue()
with block:
    with gr.Row():
        gr.Markdown("## AI Remote Horisons")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label='Your image', type="numpy")
            background = gr.Image(label='Your wushed background', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            pose = gr.Dropdown(["waving", "selfie"], label="Pose"),
            image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=1024, step=64)
            image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=1024, step=64)
            seed = gr.Number(label="Seed", precision=0)


        with gr.Column():
            result = gr.Image(label='Remote Horizons output', type='pil')
            generate_button = gr.Button("Generate Image")

    with gr.Row():
        gr.Examples(
            fn=lambda input_image, background, prompt, image_width, image_height, seed:
            generate_image(input_image, background, prompt, image_width, image_height, seed),
            examples=db_examples.examples,
            inputs=[input_image, background, prompt, image_width, image_height, seed],
            outputs=result,
            run_on_click=False
        )

    generate_button.click(fn=generate_image,
                          inputs=[input_image, background, prompt, image_width, image_height, seed],
                          outputs=result)

block.launch(server_name='127.0.0.1')