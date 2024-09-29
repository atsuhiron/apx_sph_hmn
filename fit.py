import time
import numpy as np
import scipy.optimize as so

import sph_hmn


def fit(max_apx_degree: int,
        data_arr: np.ndarray,
        phi_theta: None | np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    assert max_apx_degree >= 0
    assert data_arr.shape[0] == data_arr.shape[1]

    if phi_theta is None:
        phi_theta = sph_hmn.gen_angle_arr(len(data_arr))
    else:
        assert data_arr.shape == phi_theta.shape[0]

    opt_para = (1.77,)  # = sqrt(4π) / 2 = 0.5 / sph_harm(0, 0, θ, Φ )
    flat_arr = data_arr.flatten()
    flat_pt = phi_theta.flatten()
    for i in range(max_apx_degree + 1):
        t_start = time.time()
        para = sph_hmn.SphericalHarmonics.gen_param_with_init_value(i, opt_para)
        opt_para, cov, resi = _inner_fit(i, _get_shape(phi_theta), flat_pt, flat_arr, para)
        t_end = time.time()
        print(f"{i:<3d} residual={resi:.4e} time={t_end-t_start:.4e} [sec]")

    sph = sph_hmn.SphericalHarmonics(max_apx_degree, _get_shape(phi_theta))
    return opt_para, sph.f_split_x(phi_theta[0], phi_theta[1], *opt_para)


def _inner_fit(degree: int,
               shape: tuple[int, int, int],
               phi_theta_flatten: np.ndarray,
               y_data_flatten: np.ndarray,
               parameters: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, float]:
    sph = sph_hmn.SphericalHarmonics(degree, shape)
    para_cov = so.curve_fit(sph.f, phi_theta_flatten, y_data_flatten, p0=parameters)
    residual = np.square(y_data_flatten - sph.f(phi_theta_flatten, *tuple(para_cov[0])))
    return para_cov[0], para_cov[1], float(np.sum(residual))


def _get_shape(phi_theta: np.ndarray) -> tuple[int, int, int]:
    return phi_theta.shape[0], phi_theta.shape[1], phi_theta.shape[2]
