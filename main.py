import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch
import db_examples
from PIL import Image

# Load the Stable Diffusion XL model
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
# pipe = pipe.to("gpu")  # Use "cpu" if you don't have a GPU
torch_device = "cpu" # if torch.cuda.is_available() else "cpu"


negative_prompt = 'Deformed, mutated, extra limbs, disfigured, ugly, bad anatomy, missing limbs,' \
                  'bad, immature, cartoon, anime, painting, mutant, body horror,'  \
                                                '(six fingers), (extra fingers), (bad hands),' \
                                                '(poorly drawn hands), (fused fingers), (too many fingers),' \
                                                '(unnatural hands), (disfigured hands)'

def generate_image(input_image, background, prompt, image_width, image_height, seed):
    # Generate an image from the prompt
    color_photo = 'a color frontal photo of human'
    reccomended_prompt = '50mm portrait photography, camera lens in front, daylighting photography --beta --ar 3:2 --upbeta'
    image = pipe(prompt=color_photo + prompt + reccomended_prompt,
                 negative_prompt=negative_prompt,
                 num_inference_steps=1,
                 generator=torch.Generator(device=torch_device).manual_seed(seed)).images[0]
    return image



my_theme = gr.Theme.from_hub("JohnSmith9982/small_and_pretty")
# Launch the app
block = gr.Blocks(theme=my_theme).queue()
with block:
    with gr.Row():
        gr.Markdown("## AI Remote Horisons")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label='Your image', type='pil')
            background = gr.Image(label='Your wushed background', type='pil')
            prompt = gr.Textbox(label="Prompt")
            # negative_prompt = gr.Textbox(label="Negative prompt",
            #                              value='Deformed, mutated, extra limbs, disfigured, ugly, bad anatomy, missing limbs,' \
            #                                     'bad, immature, cartoon, anime, painting, mutant, body horror,'  \
            #                                     '(six fingers), (extra fingers), (bad hands),' \
            #                                     '(poorly drawn hands), (fused fingers), (too many fingers),' \
            #                                     '(unnatural hands), (disfigured hands)')

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
            run_on_click=False, examples_per_page=10
        )

    generate_button.click(fn=generate_image,
                          inputs=[input_image, background, prompt, image_width, image_height, seed],
                          outputs=result)

block.launch(server_name='127.0.0.1')