from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

txt = ['Three man','soccer', 'a banana']
img = Image.open('/home/work/daehyun/Photo-Editor/my_data/football.png')

inputs = processor(text=txt, images=img, return_tensors="pt", padding=True)
print(inputs.keys())
# input_ids, attention_mask: Text에 해당
# pixel_values: Image

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)
