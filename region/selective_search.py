# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import color, data, io, segmentation, transform, util, feature

#  selective search for object detection
#  input: image(numpy.array, dtype = np.uint8)
#  output: image_segmentation, dtype


def _segments(img, scale=400.0, sigma=0.8, min_size=300):
    """
        segment smallest regions by Felzenszwalb and Huttenlocher
    Examples
    ------
    >>> img = skimage.data.camera()
    >>> img = _segments(img, 1.0, 0.8, 100)
    >>> print(img.shape)
    """
    assert np.ndim(img) == 3
    img_boundary = segmentation.felzenszwalb(img, scale, sigma, min_size)
    img = np.append(img, img_boundary[:, :, np.newaxis], axis=2)
    return img


def _sim_color(r1, r2):
    """
        calculate the similarity of color(hsv)
    """
    return sum(min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"]))


def _sim_texture(r1, r2):
    """
        calculate the similarity of texture
    """
    return sum(min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"]))


def _sim_size(r1, r2, imsize):
    """
        calculate the similarity of size
        This similarity encourages small regions to merge early, 
        prevents a single region from gobbling up all other regions one by one.
    """
    return 1.0 - (r1["size"] + r2["size"])/imsize


def _sim_fill(r1, r2, imsize):
    """
        calculate the similarity of fill
        When two regions are near or one is contained by the other, 
        they are supposed to be merged early. 
    """
    bbsize = ((max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])) *
              (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"])))
    return 1.0 - (bbsize - r1["size"] - r2["size"])/imsize


def _sim(r1, r2, imsize):
    """
        calculate the similarity according to color, texture, size and fill
    """
    return (_sim_color(r1, r2) + _sim_texture(r1, r2) +
            _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _color_hist(img, bins=25):
    """
        calculate color histogram for regions
        default bins is same as "Selective Search for Object Recognition".
    """
    hist = np.array([])
    for channel in range(3):
        img_c = img[:, channel]
        hist = np.concatenate(
            [hist, np.histogram(img_c, bins=bins, range=(0.0, 255.0))[0]])
    # L1 normalization
    hist = hist / img.size
    return hist


def _texture(img):
    """
        calculate texture for regions using local binary pattern
    """
    assert img.ndim == 3
    img_tex = np.zeros_like(img)
    for channel in range(3):
        img_c = img[:, :, channel]
        img_tex[:, :, channel] = feature.local_binary_pattern(img_c, 8, 1.0)
    return img_tex


def _texture_hist(img_tex, bins=10):
    """
        calculate texture histgram for regions
        default bins is same as "Selective Search for Object Recognition".
    """
    hist = np.array([])
    for channel in range(3):
        img_texc = img_tex[:, channel]
        hist = np.concatenate(
            [hist, np.histogram(img_texc, bins=bins, range=(0.0, 255.0))[0]])
    # L1 normalization
    hist = hist / img_tex.size
    return hist


def _regions(img_seg):
    """
        construct basic region dict information
    """
    img = img_seg[:, :, :3]
    R = {}

    for y, i in enumerate(img_seg):
        for x, (r, g, b, l) in enumerate(i):
            if l not in R:
                R[l] = {"min_x": 0xffff, "min_y": 0xffff,
                        "max_x": 0, "max_y": 0, "labels": [l]}
            if x < R[l]["min_x"]:
                R[l]["min_x"] = x
            if x > R[l]["max_x"]:
                R[l]["max_x"] = x
            if y < R[l]["min_y"]:
                R[l]["min_y"] = y
            if y > R[l]["max_y"]:
                R[l]["max_y"] = y
    img_tex = _texture(img)
    for k in R.keys():
        R[k]["size"] = int((img_seg[:, :, 3] == k).sum())
        R[k]["hist_c"] = _color_hist(img[:, :][img_seg[:, :, 3] == k])
        R[k]["hist_t"] = _texture_hist(img_tex[:, :][img_seg[:, :, 3] == k])
    return R


def _neighbours(regions):
    """
        detect neighbours of all regions
    """
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False
    keys = sorted(list(regions.keys()))
    neighbours = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if intersect(regions[keys[i]], regions[keys[j]]):
                neighbours.append((keys[i], keys[j]))
    return neighbours


def _merge(r1, r2):
    """
        merge two regions
    """
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": r1["size"] + r2["size"],
        "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"])/(r1["size"] + r2["size"]),
        "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"])/(r1["size"] + r2["size"]),
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def selective_search(img, scale=400.0, sigma=0.8, min_size=300):
    assert img.shape[2] == 3, "3ch image is expected"
    img_seg = _segments(img, scale=scale, sigma=sigma, min_size=min_size)

    if img_seg is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _regions(img_seg)

    neighbours = _neighbours(R)
    S = {}
    for neighbour in neighbours:
        S[neighbour] = _sim(R[neighbour[0]],
                            R[neighbour[1]], imsize)

    while S != {}:
        (i, j), _ = sorted(list(S.items()), key=lambda a: a[1])[-1]

        # merge similar regions
        t = max(R.keys()) + 1
        R[t] = _merge(R[i], R[j])

        # remove old regions contains i or j
        keys_to_delete = []
        for k, v in S.items():
            if(i in k) or (j in k):
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del S[k]

        # calculate similarity set with the new region
        for k in filter(lambda a: a != (i, j), keys_to_delete):
            n = k[1] if k[0] in [i, j] else k[0]
            S[(t, n)] = _sim(R[t], R[n], imsize)

    regions = []
    for k, r in R.items():
        regions.append(
            {
                "rect": (r["min_x"], r["min_y"],
                         r["max_x"]-r["min_x"], r["max_y"]-r["min_y"]),
                "size": r["size"],
                "labels": r["labels"]
            }
        )
    return img_seg, regions
