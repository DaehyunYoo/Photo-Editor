# Photo-Editor

## Image Segmentation

coco dataset download

```
bash coco_download.sh
```

Train Mask R-CNN

```
python Mask_RCNN/adjust_train.py
```

### SAM
Install SAM:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

이미지 전체 영역 Segmentation:

```
python SAM/sam_generator.py
```

Prompting Segmentation:

```
python SAM/sam_predictor.py
```

### Demo page

SAM 실행 & 결과 확인:
```
streamlit run seg_app/app3.py
```

## Inpainting

## Image Generation
