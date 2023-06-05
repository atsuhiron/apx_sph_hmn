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

    def gen_m_range(self, n: int = None) -> list[int]:
        if n is None:
            n = self.max_n
        return list(range(-n, n + 1))

    def f(self, theta_phi: tuple[np.ndarray, np.ndarray], *args) -> np.ndarray:
        theta, phi = theta_phi
        arr = np.zeros_like(theta)
        arg_index = 0
        for n in self.gen_m_range():
            for m in self.gen_m_range(n):
                arr += ss.sph_harm(m, n, theta, phi).real * args[arg_index]
        return arr

    @staticmethod
    def calc_param_num(n: int) -> int:
        assert n >= 0
        return (n + 1)**2


if __name__ == "__main__":
    import plot
    p, t = _gen_angle_arr(64)

    _max_n = 3
    ar_map = {}
    for _n in range(_max_n + 1):
        sph = SphericalHarmonics(_n)
        m_range = sph.gen_m_range()
        for _m in m_range:
            mn_pair = MNPair(_m, _n)
            ar_map[mn_pair] = ss.sph_harm(_m, _n, t, p)
    plot.plot_by_degree(ar_map)

    sph = SphericalHarmonics(_max_n)
    random_amps = np.random.random(sph.calc_param_num(_max_n))
    ret = sph.f((t, p), *random_amps)
    plot.plot_tgt_apx_res(ret, ret)