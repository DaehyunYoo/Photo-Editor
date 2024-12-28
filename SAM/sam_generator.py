from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob 
import os

from sam_utils import show_anns



sam = sam_model_registry["vit_b"](checkpoint="/home/work/daehyun/Photo-Editor/SAM/sam_vit_h_4b8939.pth")
sam = sam.cuda()

mask_generator = SamAutomaticMaskGenerator(sam)


target_dir = '/home/work/daehyun/Photo-Editor/my_data'

files = glob.glob('/home/work/daehyun/Photo-Editor/{}/*.*'.format(target_dir))
os.makedirs('/home/work/daehyun/Photo-Editor/{}'.format(target_dir+'_sam'), exist_ok=True)

for fname in files:
    img = cv2.imread(fname)

    masks = mask_generator.generate(img)
    plt.imshow(img)
    show_anns(masks)
    plt.savefig(fname.replace(target_dir, target_dir+'_sam'))
    plt.close()