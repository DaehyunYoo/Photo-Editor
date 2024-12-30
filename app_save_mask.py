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
import io

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
    st.title('Segmentation with SAM')
    
    torch.cuda.empty_cache()
    with st.sidebar:
        upload = st.file_uploader('insert image',type=['png','jpg'])
        option = st.selectbox(
            'Segmentation mode',
            ('Click', 'All'))
        fore_back = st.selectbox(
            'foreground or background',
            ('foreground', 'background')
        )
        sam_button = st.button('generate mask')

        save_button = st.button('save mask')

        
    if 'image' not in st.session_state.keys():
        st.session_state['image'] = None
    if 'mask' not in st.session_state.keys():
        st.session_state['mask'] = None
    if upload:
        image = Image.open(upload).convert('RGB')
        image = ImageOps.exif_transpose(image)
        st.session_state['image'] = image

    if option == 'Click':
        if 'input_coord' not in st.session_state.keys():
            st.session_state['input_coord'] = []
            st.session_state['input_label'] = []
        if st.session_state['image'] is not None:
            print(list(st.session_state.keys()))
            h, w = st.session_state['image'].size[:2]
            
            max_sz = 350
            if max(h, w) > max_sz:
                if h > w:
                    n_h = max_sz
                    n_w = int(max_sz/h * w)
                else:
                    n_w = max_sz
                    n_h = int(max_sz / w * h)
                st.session_state['image'] = st.session_state['image'].resize((n_h,n_w))
            print(st.session_state['image'].size)
            col1, col2 = st.columns(2)
            with col1:
                if fore_back == 'foreground':
                    canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",  # 고정된 배경 색상
                            stroke_width=2,
                            stroke_color="rgba(255, 0, 0, 1)",
                            background_image=st.session_state['image'],
                            update_streamlit=True,
                            height=st.session_state['image'].size[1],
                            width=st.session_state['image'].size[0],
                            drawing_mode="point",
                            key="canvas",
                        )
                elif fore_back == 'background':
                    canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",  # 고정된 배경 색상
                            stroke_width=2,
                            stroke_color="rgba(0, 255, 0, 1)",
                            background_image=st.session_state['image'],
                            update_streamlit=True,
                            height=st.session_state['image'].size[1],
                            width=st.session_state['image'].size[0],
                            drawing_mode="point",
                            key="canvas2",
                        )

                # 클릭한 좌표 정보 가져오기
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        st.session_state['input_coord'].append(pd.json_normalize(canvas_result.json_data["objects"]))
                        st.session_state['input_label'].append(fore_back)
                        print(st.session_state['input_label'])
                        print(st.session_state['input_coord'][-1]['left'], st.session_state['input_coord'][-1]['top'])
            with col2:
                pos_points = []
                neg_points = []
                for coord, label in zip(st.session_state['input_coord'], st.session_state['input_label']):
                    if label == 'foreground':
                        pos_points.append([coord['left'].iloc[-1], coord['top'].iloc[-1]])
                    else:
                        neg_points.append([coord['left'].iloc[-1], coord['top'].iloc[-1]])

                pos_points = np.array(pos_points)
                neg_points = np.array(neg_points)
                if sam_button:
                    generator, predictor = get_sam()
                    set_pred_img(predictor, np.array(st.session_state['image']))

                    if len(neg_points) > 0:
                        input_point = np.concatenate([pos_points, neg_points], axis=0)
                        input_label = np.concatenate([np.array([1]*len(pos_points)), np.array([0]*len(neg_points))], axis=0)
                    else:
                        input_point = pos_points
                        input_label = np.array([1]*len(pos_points))
                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                        )
                    print(masks)
                    st.session_state['mask'] = masks
                    torch.cuda.empty_cache()

                    fig, ax = plt.subplots(figsize=(20,20))
                    ax.imshow(st.session_state['image'])
                    # for mask in masks:
                    show_mask(masks[0], ax)
                    ax.axis('off')
                    st.pyplot(fig)
                    
                else:
                    fig, ax = plt.subplots(figsize=(20,20))
                    ax.imshow(st.session_state['image'])
                    show_points(
                        pos_points,
                        neg_points,
                        ax)
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
                ax.imshow(image)
                show_anns(masks[0], ax)
                ax.axis('off')
                st.pyplot(fig)

    if save_button:
        if st.session_state['mask'] is not None:
            print(st.session_state['mask'].shape)
            mm = st.session_state['mask']
            # mmm = np.logical_or(np.logical_or(mm[0], mm[1]), mm[2])
            mmm = Image.fromarray(mm[0])
            img_byte_arr = io.BytesIO()
            mmm.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            # 다운로드 버튼을 만듭니다.
            st.download_button(
                label="Download Image",
                data=img_byte_arr,
                file_name='mask.jpg',
                mime='image/jpeg'
            )
             


if __name__ == '__main__':
    main()