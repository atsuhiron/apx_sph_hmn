import numpy as np

import plot
import fit


if __name__ == "__main__":
    max_n = 3
    npy_path = "your.npy"
    img = np.load(npy_path).astype(np.float64)[::2, ::2]
    apx = np.zeros_like(img, dtype=np.float64)

    for ii in range(3):
        _, apx[:, :, ii] = fit.fit(max_n, img[:, :, ii])

    plot.plot_tgt_apx_res(img.astype(np.uint8), apx.astype(np.uint8), max_n, 0, 255)
