import numpy as np
import matplotlib.pyplot as plt

from sph_hmn_param import MNPair


def _degree_to_subplot_num(mn_pair: MNPair, max_n: int) -> int:
    #      vertical    horizontal    hori shift
    return mn_pair.m * (max_n + 1) + mn_pair.n + 1


def _clip(arr: np.ndarray, v_min: int | float, v_max: int | float) -> np.ndarray:
    arr[arr < v_min] = v_min
    arr[arr > v_max] = v_max
    return arr


def plot_by_degree(array_map: dict[MNPair, np.ndarray]):
    max_n = max(map(lambda mn_pair: mn_pair.n, array_map.keys()))
    max_m_num = max_n + 1

    for mn_pair_key in array_map.keys():
        index = _degree_to_subplot_num(mn_pair_key, max_n)
        plt.subplot(max_m_num, max_n + 1, index)
        plt.title(str(mn_pair_key), fontsize=10)
        plt.imshow(array_map[mn_pair_key].real)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def plot_tgt_apx_res(target_arr: np.ndarray, apx_arr: np.ndarray,
                     max_n: int | None = None, v_min: float | None = None, v_max: float | None = None):
    if v_max is None:
        v_max = max(np.max(target_arr), np.max(apx_arr))
    if v_min is None:
        v_min = min(np.min(target_arr), np.min(apx_arr))

    plt.subplot(131)
    plt.title("target")
    plt.imshow(target_arr, vmax=v_max, vmin=v_min)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(132)
    if max_n is None:
        text = "approx"
    else:
        text = f"approx (n={max_n})"
    plt.title(text)
    plt.imshow(apx_arr, vmax=v_max, vmin=v_min)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(133)
    if target_arr.ndim == 3:
        plt.title("target - approx + 127")
        apx_arr = _clip(apx_arr, v_min, v_max)
        residual = np.int32(127) + target_arr.astype(np.int32) - apx_arr.astype(np.int32)
        residual = _clip(residual, 0, 255)
        residual = residual.astype(np.uint8)
    else:
        plt.title("target - approx")
        residual = target_arr - apx_arr
    plt.imshow(residual, vmax=v_max, vmin=v_min)
    plt.xticks([])
    plt.yticks([])

    plt.show()
