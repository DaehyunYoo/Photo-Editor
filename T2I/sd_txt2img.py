# Stable Diffusion Inference 

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)#, safety_checker=None)
pipe = pipe.to("cuda")

# prompt = "furniture, cozy, home, interior design, comfortable, warmth, relaxation, living space"

prompt = "((pedestrian:1.2)), a person walking on a busy city street, casual clothing, urban environment, clear day, dynamic scene, photorealistic, 8k resolution, high detail"

neg_prompt = "low quality, blurry, out of focus, pixelated, overexposed, underexposed"

image = pipe(prompt, negative_prompt=neg_prompt).images[0]   

# pos_prompt = "((dog)), detailed fur texture, bright eyes, happy expression, outdoors, natural lighting, photography, high-resolution, 8k, sharp focus, bokeh background"
# neg_prompt = "low quality, blurry, unrealistic, dark, low resolution, overexposed, underexposed"

# pos_prompt = "((young girl)), smiling, playful, wearing a cute dress, bright natural lighting, outdoors in a park, portrait photography, high-resolution, 8k, sharp focus, soft background blur"

# neg_prompt = "low quality, blurry, unrealistic, dark, low resolution, overexposed, underexposed, harsh shadows"

# image = pipe(pos_prompt, negative_prompt=neg_prompt).images[0]   
    
image.save("girl.png")