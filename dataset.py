import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms

path = "/DATA/bhumika1/Documents/Nikhil/pix2pix/data/data/"
# train_dataset = glob.glob(path+'/*.png')
# print(train_dataset[0])

## to see the examples in the train dataset
# for img_path in train_dataset:
#     im = Image.open(img_path)
#     np_im = np.array(im)
#     print(np_im.shape)
#     break

class AnimeDataset(Dataset):
    def __init__(self,root_dir,data):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((256,512)),transforms.Normalize(0.5,0.5)])
        self.list = glob.glob(self.root_dir+ '/'+data+'/*.png')

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self,index):
        img_path = self.list[index]
        img = Image.open(img_path)
        ten_img = self.transform(img)
        # print(ten_img.shape)
        n = ten_img.shape[2] //2
        # print(n)
        return ten_img[:,:,n:], ten_img[:,:,:n]

