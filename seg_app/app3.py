import streamlit as st
from PIL import Image
from PIL import ImageOps
from streamlit_drawable_canvas import st_canvas

import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


@st.cache_resource
def get_sam():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["vit_h"](checkpoint="/home/work/daehyun/Photo-Editor/SAM/sam_vit_h_4b8939.pth")
    sam = sam.to(device)

    generator = SamAutomaticMaskGenerator(sam)
    
    return generator

def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def main():
    st.title('welcome to the segmentation world')
    torch.cuda.empty_cache() # 메모리 비우기
    with st.sidebar:
        upload = st.file_uploader('insert image',type=['png','jpg'])
        
        sam_button = st.button('generate mask')

    if upload:
        image = Image.open(upload).convert('RGB')
        image = ImageOps.exif_transpose(image)

        image = np.array(image)

        # if 'image' not in st.session_state:
        st.session_state['image'] = image
        # max_sz = max(image.size[:2])


        st.image(st.session_state['image'])
    
    if sam_button:
        generator = get_sam()
        masks = generator.generate(st.session_state['image'])
        torch.cuda.empty_cache()

        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(image)
        show_anns(masks, ax)
        ax.axis('off')
        st.pyplot(fig)


if __name__ == '__main__':
    main()