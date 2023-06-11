import gen_data
import plot
import sph_hmn
import fit


if __name__ == "__main__":
    size = 64
    max_n = 10
    ydata = gen_data.gen_noise_2d((size, size), 1.5, 10, seed=253)
    xdata = sph_hmn.gen_angle_arr(len(ydata))

    _, apx = fit.fit(max_n, ydata)
    plot.plot_tgt_apx_res(ydata, apx, max_n)
