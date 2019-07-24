import os
import pickle
import PIL
import numpy as np
import scipy.io as sio
import skimage
import torch
import torchvision
from skimage import transform, util
from torch.utils.data import Dataset
from torchvision import transforms


class Flower17(Dataset):
    def __init__(self, jpg_dir, ground_truth_dir, regions_path, splits_path, mode="trn", split=1, transform=None):
        self.jpg_dir = jpg_dir
        self.ground_truth_dir = ground_truth_dir
        self.regions_path = regions_path
        self.splits_path = splits_path
        self.mode = mode
        self.split = split
        self.transform = transform
        splits_idx = sio.loadmat(self.splits_path)[
            '%s%d' % (self.mode, self.split)][0].tolist()
        imlist = []
        for _, _, file_paths in os.walk(self.ground_truth_dir):
            for file_path in file_paths:
                if file_path.endswith('.png'):
                    imlist.append(int(file_path.split('_')[1].split('.')[0]))
        regions_file = open(self.regions_path, 'rb')
        regions = pickle.load(regions_file)
        regions_file.close()
        self.regions = {}
        i = 0
        # a = []
        for k in list(set(splits_idx).intersection(set(imlist))):
            img_src_path = os.path.join(self.jpg_dir,  "image_%04d.jpg" % k)
            img_src = skimage.io.imread(img_src_path)
            gt_path = os.path.join(
                self.ground_truth_dir, "image_%04d.png" % k)
            gt = np.prod(skimage.io.imread(gt_path) ==
                         np.array([128, 0, 0, 255]), axis=-1)
            for r in regions[k]:
                min_x, min_y, delta_x, delta_y = r["rect"]
                gt_area = gt[min_y:(min_y+delta_y+1),
                             min_x:(min_x+delta_x+1)].sum()
                region_gt_ratio = gt_area / ((delta_x+1)*(delta_y+1))
                # ground_truth lack the 14 category
                if (region_gt_ratio > 0.2) and (region_gt_ratio < 0.5):
                    continue
                if region_gt_ratio <= 0.2 or ((delta_x+1)*(delta_y+1)) > (1.5 * gt.sum()):
                    label = 0
                elif k <= 1040:
                    label = int(((k - 1) // 80) + 1)
                else:
                    label = int((k - 1) // 80)
                self.regions[i] = {
                    "src_img": img_src_path,
                    "rect": r["rect"],
                    "label": label,
                    "gt_ratio": region_gt_ratio
                }
                i = i + 1
        #         if label not in a:
        #             a.append(label)
        #         if k == 15:
        #             print(min_x, min_y, delta_x, delta_y, label)
        # print(a)

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        assert (idx >= 0 and idx < len(self.regions)), "idx out of range"
        region = self.regions[idx]
        min_x, min_y, delta_x, delta_y = region["rect"]
        img = skimage.io.imread(region["src_img"])[
            min_y:(min_y+delta_y+1), min_x:(min_x+delta_x+1)]
        img = transform.resize(img, (227, 227))
        img = util.img_as_ubyte(img)
        img = PIL.Image.fromarray(img)
        if self.transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))])
        img = self.transform(img)
        return img, region["label"]


# train_dataset = Flower17("data/17flowers/jpg", "data/trimaps",
#                          "data/17flowers/regions.pkl", "data/datasplits.mat")
# print(len(train_dataset))
