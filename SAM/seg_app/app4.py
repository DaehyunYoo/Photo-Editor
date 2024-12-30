import streamlit as st
from PIL import Image
from PIL import ImageOps
from streamlit_drawable_canvas import st_canvas


def main():
    st.title('welcome to the segmentation world')
    with st.sidebar:
        upload = st.file_uploader('insert image',type=['png','jpg'])

    if upload:
        image = Image.open(upload)
        image = ImageOps.exif_transpose(image)
        if 'image' not in st.session_state:
            st.session_state['image'] = image
        
        h, w = st.session_state['image'].size[:2]    
        max_sz = 700
        if max(h, w) > max_sz:
            if h > w:
                n_h = max_sz
                n_w = int(max_sz/h * w)
            else:
                n_w = max_sz
                n_h = int(max_sz / w * h)
            st.session_state['image'] = st.session_state['image'].resize((n_h,n_w))
                    
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
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if objects:
                st.write("클릭한 좌표 정보:")
                for obj in objects:
                    st.json(obj)
        else:
            st.write("아직 클릭된 좌표가 없습니다.")
                         
                         


if __name__ == '__main__':
    main()