# Photo-Editor
An interactive demo that combines Segment Anything Model (SAM), GPT prompt engineering, and Stable Diffusion for intelligent image inpainting.

## Overview
This project integrates three powerful AI models to create an advanced image editing tool:
1. **SAM** (Segment Anything Model) for precise object segmentation
2. **GPT** for intelligent prompt engineering 
3. **Stable Diffusion** for high-quality image inpainting

## Features
- Interactive object selection using SAM
- Intelligent prompt generation using GPT
- High-quality image inpainting with Stable Diffusion
- User-friendly interface built with Streamlit

## Quick Start

### 1. Make Mask with SAM
Run the mask creation interface:
```bash
streamlit run app_save_mask.py
```

### 2. Image Inpainting
Runt the inpainting interface:
```bash
stramlit run app_inpaint.py
```

### Sample Results
<table>
<tr>
    <td width="33%" align="center">
        <img src="sample_result/images.jpeg" width="100%"/>
        <br>
        Original
    </td>
    <td width="33%" align="center">
        <img src="sample_result/mask.jpg" width="100%"/>
        <br>
        Mask
    </td>
    <td width="33%" align="center">
        <img src="sample_result/image_result.jpg" width="100%"/>
        <br>
        Result
    </td>
</tr>
</table>

## Advanced Features

### Image Segmentation

Download COCO dataset:
```bash
bash coco_download.sh
```

Train Mask R-CNN:
```bash
python Mask_RCNN/adjust_train.py
```

### SAM Integration

Install SAM:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Full Image Segmentation:
```bash
python SAM/sem_generator.py
```

Prompting Segmentation:
```bash
python SAM/sam_predictor.py
```

### Interactive Demo
Launch the interactive demo:
```bash
streamlit run seg_app/app3.py
```

### Stable Diffusion
Run Stable Diffusion interface:
```bash
python T2I/sd_txt2img.py
```

## Installation

1. Clone the repository:

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```