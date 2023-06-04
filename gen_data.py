import numpy as np
import scipy.ndimage as sn


def gen_noise_2d(size: tuple[int, int], power: float, max_scale: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    origin = np.random.random(size)
    noise = np.zeros(size)

    max_scale = min(max_scale, *size)
    coef_arr = np.power(np.arange(1, max_scale + 1), power).astype(np.float64)
    coef_arr /= coef_arr.sum()
    for s, p in zip(range(1, max_scale + 1), coef_arr):
        noise += sn.gaussian_filter(origin, s) * p
    return noise


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    noise_map = gen_noise_2d((64, 128), 0.2, 32)
    plt.imshow(noise_map, cmap="jet")
    plt.colorbar()
    plt.show()
