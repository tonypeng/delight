import numpy as np
import scipy.misc
import lib
import lib.data as data


def main():
    cam_data = data.load_camera_data("reflectance_illum_camera.mat")

    # first, create E, where E^k refers to E for color channel k
    basis_L, basis_R = lib.create_bases(cam_data)
    E = lib.create_E_matrices(cam_data, basis_L, basis_R)

    # define camera flash illumination coeffs
    flash_vec = 0.025 * np.ones(
        cam_data.get_reflectance_spectra()[:, 1].shape,
    )
    flash_coeffs = basis_L.T.dot(flash_vec).reshape(-1, 1)
    flash_coeffs /= np.linalg.norm(flash_coeffs)

    photo_pair = data.load_photo_pair("book.mat", flash_coeffs)

    # run the alg
    res = lib.separate_lights(photo_pair, E, 2)

    img1, img2, alpha_normalized, gamma = res['img_1'], res['img_2'], res['alpha_normalized'], res['gamma']

    scipy.misc.imsave('out/noflash.jpg', photo_pair.get_no_flash_photo())
    scipy.misc.imsave('out/flash.jpg', photo_pair.get_flash_photo())
    I_pf = np.maximum(photo_pair.get_flash_photo() - photo_pair.get_no_flash_photo(), 0)
    scipy.misc.imsave('out/pureflash.jpg', I_pf)
    scipy.misc.imsave('out/light1.jpg', img1)
    scipy.misc.imsave('out/light2.jpg', img2)
    scipy.misc.imsave('out/alpha_normalized.jpg', alpha_normalized)
    scipy.misc.imsave('out/gamma.jpg', gamma)


if __name__ == '__main__':
    main()