import numpy as np
import scipy.ndimage as sn
from PIL import Image

import plot
import fit


def to_array_rgb(path: str) -> np.ndarray:
    img_bin = Image.open(path)
    return np.asarray(img_bin.convert("RGB"))


def clip(arr: np.ndarray, v_min: int | float, v_max: int | float) -> np.ndarray:
    arr[arr < v_min] = v_min
    arr[arr > v_max] = v_max
    return arr


def apx_monochrome(arr: np.ndarray, max_n: int, sigma: int | float):
    if arr.ndim == 3:
        arr = np.mean(arr.astype(np.float64), axis=2)
    _, apx = fit.fit(max_n, sn.gaussian_filter(arr, sigma))
    apx = clip(apx, 0, 255)
    plot.plot_tgt_apx_res(arr, apx, max_n)


def apx_rgb(arr: np.ndarray, max_n: int, sigma: int | float):
    apx = np.zeros_like(arr, dtype=np.float64)
    for ii in range(3):
        _, apx[:, :, ii] = fit.fit(max_n, sn.gaussian_filter(arr[:, :, ii], sigma))
    apx = clip(apx, 0, 255)
    plot.plot_tgt_apx_res(arr.astype(np.uint8), apx.astype(np.uint8), max_n, 0, 255)


if __name__ == "__main__":
    npy_path = "your.npy"
    img = np.load(npy_path).astype(np.float64).mean(axis=2)[::6, ::6]

    apx_rgb(img, max_n=8, sigma=1.5)
