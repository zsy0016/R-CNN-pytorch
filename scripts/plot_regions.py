import pickle
import skimage
import numpy as np
from skimage import io, draw, segmentation, color
import matplotlib.pyplot as plt


def plot_regions(img_path, regions):
    img = io.imread(img_path)
    img_region = img.copy()
    for region in regions:
        min_x, min_y, delta_x, delta_y = region["rect"]
        rr, cc = draw.line(min_y, min_x, min_y+delta_y, min_x)
        draw.set_color(img_region, [rr, cc], [255, 0, 0])
        rr, cc = draw.line(min_y, min_x+delta_x, min_y+delta_y, min_x+delta_x)
        draw.set_color(img_region, [rr, cc], [255, 0, 0])
        rr, cc = draw.line(min_y, min_x, min_y, min_x+delta_x)
        draw.set_color(img_region, [rr, cc], [255, 0, 0])
        rr, cc = draw.line(min_y+delta_y, min_x, min_y+delta_y, min_x+delta_x)
        draw.set_color(img_region, [rr, cc], [255, 0, 0])
    img_boundary = segmentation.felzenszwalb(img, 500, 0.8, 200)
    img_boundary = segmentation.mark_boundaries(img, img_boundary)
    plt.subplots(133)
    plt.subplot(131)
    io.imshow(img)
    plt.axis('off')
    plt.subplot(132)
    io.imshow(img_region)
    plt.axis('off')
    plt.subplot(133)
    io.imshow(img_boundary)
    plt.axis('off')
    plt.show()


def main():
    img_path = 'data/17flowers/jpg/image_0170.jpg'
    region_path = 'data/17flowers/regions.pkl'

    with open(region_path, 'rb') as f:
        regions = pickle.load(f)[170]
        plot_regions(img_path, regions)


if __name__ == "__main__":
    main()
