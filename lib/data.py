import scipy.io


class CameraData:
    def __init__(self, cam_response_spectra, illumination_spectra, reflectance_spectra):
        self._C = cam_response_spectra
        self._L = illumination_spectra
        self._R = reflectance_spectra

    def get_camera_response_spectra(self):
        return self._C

    def get_illumination_spectra(self):
        return self._L

    def get_reflectance_spectra(self):
        return self._R


class PhotoPair:
    def __init__(self, noflash, flash, flash_coeffs):
        self._I_nf = noflash
        self._I_f = flash
        self._flash_coeffs = flash_coeffs

    def get_no_flash_photo(self):
        return self._I_nf

    def get_flash_photo(self):
        return self._I_f

    def get_flash_coeffs(self):
        return self._flash_coeffs


def load_camera_data(path):
    mat = scipy.io.loadmat(path)
    return CameraData(mat['C'], mat['L'], mat['R'])


def load_photo_pair(path, flash_coeffs):
    mat = scipy.io.loadmat(path)
    return PhotoPair(mat['im_nf'], mat['im_f'], flash_coeffs)
