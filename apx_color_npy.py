from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.ndimage as sn
from PIL import Image

import plot
import fit


def to_array_rgb(path: str) -> np.ndarray:
    img_bin = Image.open(path)
    return np.asarray(img_bin.convert("RGB"))


def apx_monochrome(arr: np.ndarray, max_n: int, sigma: int | float,
                   pre_params: np.ndarray | None = None, min_n: int | None = None) -> np.ndarray:
    if arr.ndim == 3:
        arr = np.mean(arr.astype(np.float64), axis=2)

    blured = np.zeros_like(arr, dtype=np.float64)
    if sigma != 0:
        blured = sn.gaussian_filter(arr, sigma)
    else:
        blured = arr

    opt_param, apx = fit.fit(max_n, blured, pre_params=pre_params, min_apx_degree=min_n)
    plot.plot_tgt_apx_res(blured, apx, max_n)
    return opt_param


def apx_rgb(arr: np.ndarray, max_n: int, sigma: int | float,
            pre_params: np.ndarray | None = None, min_n: int | None = None) -> np.ndarray:
    assert pre_params is None or pre_params.ndim == 2
    if pre_params is None:
        pre_params = [None, None, None]

    apx = np.zeros_like(arr, dtype=np.float64)
    blured = np.zeros_like(arr, dtype=np.float64)
    if sigma != 0:
        for ii in range(3):
            blured[:, :, ii] = sn.gaussian_filter(arr[:, :, ii], sigma)
    else:
        blured = arr

    opt_params = [None, None, None]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for ii in range(3):
            future = executor.submit(fit.fit_with_idx, idx=ii, max_apx_degree=max_n, data_arr=blured[:, :, ii],
                                     pre_params=pre_params[ii], min_apx_degree=min_n)
            futures.append(future)
        for future in as_completed(futures):
            result = future.result()
            apx[:, :, int(result[0])] = result[2]
            opt_params[int(result[0])] = result[1]

    plot.plot_tgt_apx_res(blured.astype(np.uint8), apx.astype(np.uint8), max_n, 0, 255)
    return np.array(opt_params)


if __name__ == "__main__":
    npy_path = "your.npy"
    img = np.load(npy_path).astype(np.float64).mean(axis=2)[::6, ::6]

    apx_rgb(img, max_n=8, sigma=1.5)
