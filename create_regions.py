import os
import sys
import pickle
import argparse
from skimage import io, transform, util
from region import selective_search


def main():
    parser = argparse.ArgumentParser("parser for creating regions")
    parser.add_argument("--data_dir", type=str, default="data/17flowers/jpg")
    parser.add_argument("--save_path", type=str,
                        default="data/17flowers/regions.pkl")
    args = parser.parse_args()
    data_dir = args.data_dir
    save_path = args.save_path

    regions = {}
    for _, _, img_paths in os.walk(data_dir):
        for img_path in img_paths:
            if img_path.endswith(".jpg"):
                img = io.imread(os.path.join(data_dir, img_path))
                img_index = int(img_path.split('_')[1].split('.')[0])
                regions[img_index] = selective_search(img)[1]
                print(img_path, len(regions[img_index]))

    with open(save_path, "wb") as f:
        pickle.dump(regions, f)


if __name__ == "__main__":
    main()
