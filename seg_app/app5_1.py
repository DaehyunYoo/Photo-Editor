import streamlit as st
from PIL import Image
from PIL import ImageOps
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
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
    predictor = SamPredictor(sam)
    return generator, predictor

def set_pred_img(predictor, img):
    predictor.set_image(img)

def show_points(pos_points, neg_points, ax, marker_size=700):
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

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
    
    torch.cuda.empty_cache()
    with st.sidebar:
        upload = st.file_uploader('insert image',type=['png','jpg'])
        option = st.selectbox(
            'Segmentation mode',
            ('Click', 'All'))
        sam_button = st.button('generate mask')
        
    if 'image' not in st.session_state:
        st.session_state['image'] = None
    if upload:
        image = Image.open(upload).convert('RGB')
        image = ImageOps.exif_transpose(image)
        st.session_state['image'] = image

    if option == 'Click':
        if 'input_coord' not in st.session_state:
            st.session_state['input_coord'] = []
        if st.session_state['image'] is not None:
            h, w = st.session_state['image'].size
            
            max_sz = 350
            if max(h, w) > max_sz:
                if h > w:
                    n_h = max_sz
                    n_w = int(max_sz/h * w)
                else:
                    n_w = max_sz
                    n_h = int(max_sz/w * h)
                st.session_state['image'] = st.session_state['image'].resize((n_h, n_w))
            
            col1, col2 = st.columns(2)
            with col1:
                # Display the image first
                st.image(st.session_state['image'], use_column_width=True)
                
                # Then create the canvas
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="rgba(255, 0, 0, 1)",
                    background_color="rgba(255, 255, 255, 0.0)",
                    height=st.session_state['image'].size[1],
                    width=st.session_state['image'].size[0],
                    drawing_mode="point",
                    key="canvas",
                )
                
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        st.session_state['input_coord'].append(pd.json_normalize(canvas_result.json_data["objects"]))

            with col2:
                if st.session_state['input_coord']:
                    pos_points = []
                    for coord in st.session_state['input_coord']:
                        pos_points.append([coord['left'].iloc[-1], coord['top'].iloc[-1]])
                    pos_points = np.array(pos_points)
                    
                    if sam_button and len(pos_points) > 0:
                        generator, predictor = get_sam()
                        set_pred_img(predictor, np.array(st.session_state['image']))
                        
                        input_point = pos_points
                        input_label = np.array([1] * len(pos_points))
                        
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=True,
                        )
                        
                        torch.cuda.empty_cache()

                        fig, ax = plt.subplots(figsize=(20,20))
                        ax.imshow(st.session_state['image'])
                        for mask in masks:
                            show_mask(mask, ax)
                        ax.axis('off')
                        st.pyplot(fig)
                    
    elif option == 'All':
        if st.session_state['image'] is not None:
            st.image(st.session_state['image'])
            if sam_button:
                generator, predictor = get_sam()
                masks = generator.generate(np.array(st.session_state['image']))
                torch.cuda.empty_cache()

                fig, ax = plt.subplots(figsize=(20,20))
                ax.imshow(st.session_state['image'])
                show_anns(masks, ax)
                ax.axis('off')
                st.pyplot(fig)

if __name__ == '__main__':
    main()