from concurrent.futures import ThreadPoolExecutor, as_completed

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

    blured = np.zeros_like(arr, dtype=np.float64)
    if sigma != 0:
        blured = sn.gaussian_filter(arr, sigma)
    else:
        blured = arr

    _, apx = fit.fit(max_n, blured)
    apx = clip(apx, 0, 255)
    plot.plot_tgt_apx_res(blured, apx, max_n)


def apx_rgb(arr: np.ndarray, max_n: int, sigma: int | float):
    apx = np.zeros_like(arr, dtype=np.float64)
    blured = np.zeros_like(arr, dtype=np.float64)
    if sigma != 0:
        for ii in range(3):
            blured[:, :, ii] = sn.gaussian_filter(arr[:, :, ii], sigma)
    else:
        blured = arr

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for ii in range(3):
            future = executor.submit(fit.fit_with_idx, idx=ii, max_apx_degree=max_n, data_arr=blured[:, :, ii])
            futures.append(future)
        for future in as_completed(futures):
            result = future.result()
            apx[:, :, int(result[0])] = result[2]

    apx = clip(apx, 0, 255)
    plot.plot_tgt_apx_res(blured.astype(np.uint8), apx.astype(np.uint8), max_n, 0, 255)


if __name__ == "__main__":
    npy_path = "your.npy"
    img = np.load(npy_path).astype(np.float64).mean(axis=2)[::6, ::6]

    apx_rgb(img, max_n=8, sigma=1.5)
