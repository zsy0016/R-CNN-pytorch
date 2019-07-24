import argparse
import os
from collections import OrderedDict, Counter

import matplotlib.pyplot as plt
import numpy as np
import PIL
import skimage
import torch
import torch.distributed as dist
import torchvision
from skimage import io, transform, util, draw
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

import cnn
from region import selective_search

torch.multiprocessing.set_start_method('spawn')


def nms(regions):
    def iou(a, b):
        img = np.zeros([max(a["max_y"], b["max_y"]),
                        max(a["max_x"], b["max_x"]), 2])
        img[a["min_y"]:(a["max_y"]+1), a["min_x"]:(a["max_x"]+1), 0] = 1
        img[b["min_y"]:(b["max_y"]+1), b["min_x"]:(b["max_x"]+1), 0] = 1
        img_intersect = np.prod(img, axis=-1)
        return img_intersect.sum() / (img[:, :, 0].sum() + img[:, :, 1].sum() - img_intersect.sum())

    def merge(a, b):
        assert a["category"] == b["category"], "Try to merge regions of 2 categories"
        a["min_x"] = min(a["min_x"], b["min_x"])
        a["min_y"] = min(a["min_y"], b["min_y"])
        a["max_x"] = max(a["max_x"], b["max_x"])
        a["max_y"] = max(a["max_y"], b["max_y"])

    regions = sorted(regions, key=lambda a: a["probability"], reverse=True)
    region = regions[0]
    regions = regions[1:]
    merge_regions = []
    for r in regions:
        if r["category"] == region["category"] and iou(r, region) > 0.5:
            merge(region, r)
    return region

def plot_img(img, region):
    img_region = img.copy()
    min_x = region["min_x"]
    min_y = region["min_y"]
    max_x = region["max_x"]
    max_y = region["max_y"]
    rr, cc = draw.line(min_y, min_x, max_y+1, min_x)
    draw.set_color(img_region, [rr, cc], [255, 0, 0])
    rr, cc = draw.line(min_y, max_x+1, max_y+1, max_x+1)
    draw.set_color(img_region, [rr, cc], [255, 0, 0])
    rr, cc = draw.line(min_y, min_x, min_y, max_x+1)
    draw.set_color(img_region, [rr, cc], [255, 0, 0])
    rr, cc = draw.line(max_y+1, min_x, max_y+1, max_x+1)
    draw.set_color(img_region, [rr, cc], [255, 0, 0])
    return img_region



def main():
    parser = argparse.ArgumentParser("parser for testing R-CNN on 17flowers")
    parser.add_argument("--image", type=str,
                        default="data/17flowers/jpg/image_0045.jpg")
    parser.add_argument("--cnn", type=str, default="alexnet")
    parser.add_argument("--model_path", type=str, default="model/alexnet.pkl")
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()

    assert args.cnn in ["alexnet", "vgg"], "unknown cnn model"
    if args.cnn == "alexnet":
        model = cnn.Alexnet()
    if args.cnn == "vgg":
        model = cnn.Vgg()
    if args.cuda:
        model.cuda()
    trained_net = torch.load(args.model_path)
    model.load_state_dict(trained_net)
    model.eval()

    img = io.imread(args.image)
    _, regions = selective_search(img)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))])
    region_imgs = []
    for r in regions:
        min_x, min_y, delta_x, delta_y = r["rect"]
        region_img = img[min_y:(min_y+delta_y+1), min_x:(min_x+delta_x+1)]
        region_img = skimage.transform.resize(region_img, (227, 227))
        region_img = util.img_as_ubyte(region_img)
        region_img = PIL.Image.fromarray(region_img)
        region_imgs.append(transform(region_img).unsqueeze(dim=0))
    data = torch.cat(region_imgs, dim=0)
    if args.cuda:
        data = data.cuda()
    y_pred = torch.nn.Softmax(dim=-1)(model(data))
    if args.cuda:
        y_pred = y_pred.detach().cpu()
    y_pred = y_pred.numpy()
    y_pred_category = np.argmax(y_pred, axis=-1)

    select_regions = []
    for i in range(len(regions)):
        if y_pred_category[i] == 0:
            continue
        elif y_pred[i, y_pred_category[i]] < 0.5:
            continue
        else:
            min_x, min_y, delta_x, delta_y = regions[i]["rect"]
            select_regions.append(
                {"min_x": min_x,
                 "min_y": min_y,
                 "max_x": min_x + delta_x,
                 "max_y": min_y + delta_y,
                 "probability": y_pred[i, y_pred_category[i]],
                 "category": y_pred_category[i]}
            )
    category = Counter([r["category"]
                        for r in select_regions]).most_common(1)[0][0]
    reserved_regions = []
    for r in select_regions:
        if r["category"] == category:
            reserved_regions.append(r)
    nms_region = nms(reserved_regions)

    flower_dict = {
        0: "Background", 
        1: "Daffodil",
        2: "Snowdrop",
        3: "Bluebell",
        4: "Crocus",
        5: "Iris",
        6: "Tigerlily",
        7: "Tulip",
        8: "Fritillary",
        9: "Sunflower",
        10: "Daisy",
        11: "Colts'Foot",
        12: "Dandelion",
        13: "Cowslip",
        14: "Buttercup",
        15: "Windflower",
        16: "Pansy"
    }
    img_region = plot_img(img, nms_region)
    plt.subplots(122)
    plt.subplot(121)
    io.imshow(img)
    plt.axis('off')
    plt.subplot(122)
    io.imshow(img_region)
    plt.text(nms_region["min_x"]+20, nms_region["min_y"]+20, flower_dict[category], color="r")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
