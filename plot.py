import numpy as np
import matplotlib.pyplot as plt

from sph_hmn_param import MNPair


def _degree_to_subplot_num(mn_pair: MNPair, max_n: int) -> int:
    #      vertical    horizontal    hori shift
    return mn_pair.m * (max_n + 1) + mn_pair.n + 1


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


def plot_tgt_apx_res(target_arr: np.ndarray, apx_arr: np.ndarray, max_n: int | None = None):
    v_max = max(np.max(target_arr), np.max(apx_arr))
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
    plt.title("target - approx")
    plt.imshow(target_arr - apx_arr, vmax=v_max, vmin=v_min)
    plt.xticks([])
    plt.yticks([])

    plt.show()
