import streamlit as st
from PIL import Image
from PIL import ImageOps
from streamlit_drawable_canvas import st_canvas

import numpy as np
import pandas as pd
import cv2
import torch
from matplotlib import pyplot as plt
from diffusers import StableDiffusionInpaintPipeline

import openai
import os


@st.cache_resource
def get_sd():
    pipe = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting',
                                                    #   revision='fp16',
                                                      torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    return pipe

@st.cache_resource
def get_assist():
    api_key = os.environ['OPENAI_API_KEY']

    client = openai.OpenAI(api_key=api_key)
    assistant = client.beta.assistants.retrieve(assistant_id='asst_DPIWgRQEsfg4hilmf1pj8bim')
    # print(assistant)

    thread = client.beta.threads.create()

    return client, thread, assistant


def main():
    st.title('Inpainting with Stable Diffusion')
    
    torch.cuda.empty_cache()
    with st.sidebar:
        upload = st.file_uploader('insert image',type=['png','jpg'])
        upload2 = st.file_uploader('insert mask',type=['png','jpg'])

        inpaint_button = st.button('Remove')
        edit_button = st.button('Edit')

        prompt = st.text_input("Write the text here")
            
        print("from text_input", prompt)
        st.session_state['prompt'] = prompt

        button_gen = st.button('generate image with text')

        
    if upload:
        image = Image.open(upload).convert('RGB')
        image = ImageOps.exif_transpose(image)
        st.session_state['image'] = image

        h, w = st.session_state['image'].size[:2]
        
        max_sz = 512
        if h > w:
            n_h = max_sz
            n_w = int(max_sz/h * w)
        else:
            n_w = max_sz
            n_h = int(max_sz / w * h)
        st.session_state['image'] = st.session_state['image'].resize((n_h,n_w))
        print(st.session_state['image'].size)

        if st.session_state['image'] is not None:
            st.image(st.session_state['image'])

    if upload2:
        mask = Image.open(upload2).convert('L')
        mask = ImageOps.exif_transpose(mask)
        st.session_state['mask'] = mask

        h, w = st.session_state['mask'].size[:2]
        
        max_sz = 512
        if h > w:
            n_h = max_sz
            n_w = int(max_sz/h * w)
        else:
            n_w = max_sz
            n_h = int(max_sz / w * h)
        st.session_state['mask'] = st.session_state['mask'].resize((n_h,n_w))
        print(st.session_state['mask'].size)
        if st.session_state['mask'] is not None:
            st.image(st.session_state['mask'])

    if inpaint_button:
        if st.session_state['image'] is not None and st.session_state['mask'] is not None:
            pipe = get_sd()
            prompt = 'background'
            output = pipe(prompt=prompt, image=st.session_state['image'], mask_image=st.session_state['mask']).images[0]
            st.image(output)
    
    if button_gen:
        print("input_prompt:", st.session_state['prompt'])
        pipe = get_sd()
        client, thread, assistant = get_assist()

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=st.session_state['prompt']
        )
        
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
        
        if run.status == 'completed': 
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            prompt_assisted = messages.data[0].content[0].text.value
            print("Received prompt:", prompt_assisted)  # 디버깅을 위한 출력 추가
            
            # 더 안전한 prompt 파싱 로직
            try:
                if 'positive prompt:' in prompt_assisted.lower() and 'negative prompt:' in prompt_assisted.lower():
                    pos_prompt = prompt_assisted.lower().split('positive prompt:')[1].split('negative prompt:')[0].strip()
                    neg_prompt = prompt_assisted.lower().split('negative prompt:')[1].strip()
                else:
                    # Assistant가 예상된 형식으로 응답하지 않은 경우의 폴백 처리
                    pos_prompt = st.session_state['prompt']
                    neg_prompt = ""
                    
                print("Parsed prompts:", pos_prompt, "|||", neg_prompt)  # 디버깅을 위한 출력 추가
                
                output = pipe(
                    prompt=pos_prompt, 
                    negative_prompt=neg_prompt, 
                    image=st.session_state['image'], 
                    mask_image=st.session_state['mask']
                ).images[0]
                st.image(output)
            except Exception as e:
                st.error(f"Error parsing prompt: {str(e)}")
                print(f"Error details: {str(e)}")  # 디버깅을 위한 출력 추가
        else:
            st.write(f"Run status: {run.status}")


if __name__ == '__main__':
    main()