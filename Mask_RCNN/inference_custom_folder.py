import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import v2
import torchvision.transforms as T
from torchvision import tv_tensors, datasets


from torchvision.utils import save_image
from torchinfo import summary
from torchvision.utils import draw_segmentation_masks

import glob
import os


transforms_test = T.Compose(
    [
        T.ToTensor(),
    ]
)

class CustomDataset(Dataset):
    def __init__(self, files, transforms):
        self.files = files
        self.transforms = transforms

    def __getitem__(self,idx):
        fname = self.files[idx]
        img = Image.open(fname)
        inputs = self.transforms(img)
        return inputs, fname

    def __len__(self):
        return len(self.files)


target_dir = 'my_data'

files = glob.glob('/home/work/daehyun/Photo-Editor/{}/*.*'.format(target_dir))
os.makedirs('/home/work/daehyun/Photo-Editor/{}'.format(target_dir+'_result'), exist_ok=True)
testset = CustomDataset(files, transforms_test)

testloader = DataLoader(testset, batch_size=1, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = maskrcnn_resnet50_fpn(pretrained=True, progress=False).to(device)
summary(model)

model.eval()
loader = tqdm(testloader)
with torch.no_grad():
    for i, (images, fname) in enumerate(loader):
        if i == 10:
            break
        images = images.to(device)
        outputs = model(images)
        candidate_idx = torch.where(outputs[0]['scores'] > .5)[0]
        a = draw_segmentation_masks(images[0], torch.where(outputs[0]['masks'][candidate_idx].squeeze(1)>0.8, 1, 0).type(torch.BoolTensor))
        save_image(a,fname[0].replace(target_dir, target_dir+'_result'))
    


        









