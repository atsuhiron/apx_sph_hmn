import numpy as np
import matplotlib.pyplot as plt

from sph_hmn_param import MNPair


def _degree_to_subplot_num(mn_pair: MNPair, max_n: int) -> int:
    return mn_pair.m * (max_n + 1) + mn_pair.n + 1


def plot_by_degree(array_map: dict[MNPair, np.ndarray]):
    max_n = max(map(lambda mn_pair: mn_pair.n, array_map.keys()))
    max_m_num = max_n + 1

    for mn_pair_key in array_map.keys():
        index = _degree_to_subplot_num(mn_pair_key, max_n)
        plt.subplot(max_m_num, max_n + 1, index)
        plt.title(str(mn_pair_key))
        plt.imshow(array_map[mn_pair_key].real)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()
