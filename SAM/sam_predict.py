from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob 
import os

from sam_utils import show_box, show_mask, show_points


sam = sam_model_registry["vit_b"](checkpoint="/home/work/daehyun/Photo-Editor/SAM/sam_vit_h_4b8939.pth")
sam = sam.cuda()

predictor = SamPredictor(sam)


# sample image (shushi)
fname = '/home/work/daehyun/Photo-Editor/my_data/baseball.jpeg'

img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
predictor.set_image(img)

print(predictor.features.shape)

input_point = np.array([[1500, 2000]])  # points

input_label = np.array([1])  # 1: foreground / 0: background

plt.figure(figsize=(10,10))
plt.imshow(img)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.savefig('/home/work/daehyun/Photo-Editor/my_data/baseball_point.png')
plt.close()

masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                        )

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig('point.png')
    plt.close()

# input_point: x, y 좌표 / input_label: point 하나가 foreground인지 제외되어야하는 background인지. 두 개 개수가 동일해야 에러가 발생하지 않음. 
input_point = np.array([[1500,3000]])  # points

input_label = np.array([1])  # 1: foreground / 0: background

plt.figure(figsize=(10,10))
plt.imshow(img)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.savefig('/home/work/daehyun/Photo-Editor/SAM/point.png')
plt.close()

masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                        )

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig('point.png')
    plt.close()


input_point = np.array([[1500, 2000],[1350,2300]])  # points

input_label = np.array([1,0])  # 1: foreground / 0: background

plt.figure(figsize=(10,10))
plt.imshow(img)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.savefig('/home/baekw92/point.png')
plt.close()

masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                        )

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig('point.png')
    plt.close()