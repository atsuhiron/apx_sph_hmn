import numpy as np
import scipy.special as ss

from sph_hmn_param import MNPair


def _gen_angle_arr(theta_split_num: int) -> tuple[np.ndarray, np.ndarray]:
    phi_split_num = 2 * theta_split_num
    theta_arr = np.linspace(0, np.pi, theta_split_num)
    phi_arr = np.linspace(0, 2*np.pi, phi_split_num)
    return np.meshgrid(phi_arr, theta_arr)


class SphericalHarmonics:
    def __init__(self, max_n: int):
        assert max_n >= 0
        self.max_n = max_n

    def gen_m_range(self) -> list[int]:
        return list(range(self.max_n + 1))


if __name__ == "__main__":
    import plot
    p, t = _gen_angle_arr(64)

    _max_n = 4
    ar_map = {}
    for n in range(_max_n + 1):
        sph = SphericalHarmonics(n)
        m_range = sph.gen_m_range()
        for m in m_range:
            mn_pair = MNPair(m, n)
            ar_map[mn_pair] = ss.sph_harm(m, n, t, p)
    plot.plot_by_degree(ar_map)