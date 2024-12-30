# Inpainting
# Mask는 SAM_predictor를 사용해서 생성

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from PIL import ImageOps
import torch
import numpy as np
import cv2

pipe = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting',
                                                      revision='fp16',
                                                      torch_dtype=torch.float16)
pipe = pipe.to('cuda')
prompt='empty'
# prompt = 'a yellow one-seater sofa'
# prompt = '"A cozy, modern one-seater sofa in the empty space next to the window in a contemporary living room. The sofa is light grey and made of soft fabric, complementing the minimalistic decor of the room."'
# prompt = "Inpainting a contemporary styled large sofa in the center of a modern living room. The sofa should have deep, comfortable cushions and be made of light grey velvet fabric to harmonize with the room's interior."
# prompt = "A cozy, modern {one-seater|single-seater} sofa in the empty space next to the window in a contemporary living room. The sofa is ((light grey)) and made of soft fabric, (complementing the minimalistic decor of the room), without any other furniture around it."

image = Image.open('Image.png')
image = ImageOps.exif_transpose(image)
image = image.resize((512,512))
mask_image = Image.open('mask.png')

kernel = np.ones((3, 3), np.uint8)
mask_image = cv2.dilate(np.array(mask_image), kernel, iterations=5)  #// make dilation image
mask_image = cv2.resize(mask_image, (512,512))
image = pipe(prompt=prompt, image=image, mask_image=Image.fromarray(mask_image)).images[0]

image.save('inpainted.png')