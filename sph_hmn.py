import numpy as np
import scipy.special as ss

from sph_hmn_param import MNPair


def gen_angle_arr(split_num: int) -> np.ndarray:
    theta_arr = np.linspace(0, np.pi, split_num)
    phi_arr = np.linspace(0, np.pi, split_num)
    return np.array(np.meshgrid(phi_arr, theta_arr))


class SphericalHarmonics:
    def __init__(self, max_n: int, phi_theta_shape: tuple[int, int, int]):
        assert max_n >= 0
        assert phi_theta_shape[0] == 2
        self.max_n = max_n
        self.shape = phi_theta_shape

    def gen_m_range(self, n: int = None) -> list[int]:
        if n is None:
            n = self.max_n
        return list(range(n + 1))

    def f_split_x(self, phi: np.ndarray, theta: np.ndarray, *args) -> np.ndarray:
        arr = np.zeros_like(theta)
        arg_index = 0
        for n in range(self.max_n + 1):
            for m in self.gen_m_range(n):
                arr += ss.sph_harm(m, n, theta, phi).real * args[arg_index]
                arg_index += 1
        return arr

    def f(self, phi_theta: np.ndarray, *args) -> np.ndarray:
        pt = phi_theta.reshape(self.shape)
        return self.f_split_x(pt[0], pt[1], *args).ravel()

    @staticmethod
    def calc_param_num(n: int) -> int:
        assert n >= 0
        return ((n + 1) * (n + 2)) // 2

    @staticmethod
    def gen_param_with_init_value(degree: int, init_param: tuple[float, ...] | np.ndarray) -> tuple[float, ...]:
        param_num = SphericalHarmonics.calc_param_num(degree)
        assert param_num >= len(init_param)

        param_arr = np.zeros(param_num, dtype=np.float64)
        param_arr[:len(init_param)] = init_param
        return tuple(param_arr)


if __name__ == "__main__":
    import plot
    p, t = gen_angle_arr(64)

    _max_n = 3
    ar_map = {}
    for _n in range(_max_n + 1):
        sph = SphericalHarmonics(_n, (2, 64, 64))
        m_range = sph.gen_m_range()
        for _m in m_range:
            mn_pair = MNPair(_m, _n)
            ar_map[mn_pair] = ss.sph_harm(_m, _n, t, p)
    plot.plot_by_degree(ar_map)

    sph = SphericalHarmonics(_max_n, (2, 64, 64))
    random_amps = np.random.random(sph.calc_param_num(_max_n))
    ret = sph.f_split_x(p, t, *random_amps)
    plot.plot_tgt_apx_res(ret, ret)
