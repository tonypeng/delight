import math
import numpy as np
import scipy.sparse.linalg
import scipy.linalg

from .data import CameraData, PhotoPair


def create_bases(cam_data: CameraData):
    L = cam_data.get_illumination_spectra()
    R = cam_data.get_reflectance_spectra()

    UR, _, _ = scipy.sparse.linalg.svds(R, k=3)
    UL, _, _ = scipy.sparse.linalg.svds(L, k=3)

    return scipy.linalg.orth(UL), scipy.linalg.orth(UR)


def create_E_matrices(cam_data: CameraData, basis_L, basis_R):
    E = []
    C = cam_data.get_camera_response_spectra()
    for i in range(3):
        E.append(basis_R.T.dot(np.diag(C[:, i])).dot(basis_L))
    return np.array(E)


def separate_lights(photo_pair: PhotoPair, E, N):
    f = photo_pair.get_flash_coeffs()
    I_nf = photo_pair.get_no_flash_photo()
    I_f = photo_pair.get_flash_photo()
    I_pf = np.maximum(I_f - I_nf, 0)

    # reshape to 4d tensors (nx1 column vector at each height * width)
    I_nf = I_nf.reshape(I_nf.shape + (1,))
    I_pf = I_pf.reshape(I_pf.shape + (1,))

    # step 1. solve for alpha(p) = eta_f(p) * a(p)
    E_f_T = np.matmul(E, f).reshape((3, 3))  # axis 0: channels, axis 1: reflectance basis
    alpha = np.linalg.solve(E_f_T, I_pf)

    # step 2: calculate beta/gamma
    alpha_norms = np.linalg.norm(alpha, axis=2)
    alpha_norms = alpha_norms.reshape(alpha_norms.shape + (1,))

    alpha_normalized = alpha / alpha_norms
    alpha_normalized_T = np.transpose(alpha_normalized, (0, 1, 3, 2))

    alpha_normalized_T_E_0 = np.matmul(alpha_normalized_T, E[0, :, :])
    alpha_normalized_T_E_1 = np.matmul(alpha_normalized_T, E[1, :, :])
    alpha_normalized_T_E_2 = np.matmul(alpha_normalized_T, E[2, :, :])

    alpha_normalized_T_E = np.concatenate((alpha_normalized_T_E_0, alpha_normalized_T_E_1, alpha_normalized_T_E_2),
                                          axis=2)

    beta = np.linalg.solve(alpha_normalized_T_E, I_nf)
    beta_norms = np.linalg.norm(beta, axis=2)
    beta_norms_orig = beta_norms  # keep this shape for later
    beta_norms = beta_norms.reshape(beta_norms.shape + (1,))
    gamma = beta / beta_norms

    if N == 2:
        # reshape gamma into a list of (3,) arrays for ransac
        gamma_lst = np.reshape(gamma, (-1, 3))
        # 3. estimate light basis vectors by using ransac to find an arc on S^2
        b_light1, b_light2 = ransac_sphere_arc(gamma_lst, 0.3 / 180.0 * math.pi)
        B_mat = np.array([b_light1, b_light2])

        def _solve_lstsq(g):
            z, _, _, _ = np.linalg.lstsq(B_mat.T, g)
            return np.maximum(z, 0)

        # 4. Solve for z
        # np.linalg.lstsq is not broadcastable :(
        # TODO: look into vectorized SVD lstsq
        z = np.array(
            [np.array([_solve_lstsq(g) for g in g_h]) for g_h in gamma]
        )

        img_1 = calculate_separated_image(beta_norms_orig, alpha, E, z[:, :, 0, :], b_light1)
        img_2 = calculate_separated_image(beta_norms_orig, alpha, E, z[:, :, 1, :], b_light2)

        return {
            'img_1': img_1,
            'img_2': img_2,
            'alpha_normalized': alpha_normalized.reshape(img_1.shape),
            'gamma': gamma.reshape(img_1.shape),
        }
    else:
        raise NotImplementedError('N=' + N + ' is not supported.')


def calculate_separated_image(beta_norms, alpha, E, est_shadings, est_b):
    alpha_T = np.transpose(alpha, (0, 1, 3, 2))
    I_0 = np.matmul(np.matmul(alpha_T, E[0, :, :]), est_b)
    I_1 = np.matmul(np.matmul(alpha_T, E[1, :, :]), est_b)
    I_2 = np.matmul(np.matmul(alpha_T, E[2, :, :]), est_b)
    return beta_norms * est_shadings * np.concatenate((I_0, I_1, I_2), axis=2)


def ransac_sphere_arc(points, inlier_angle_rad_threshold, iterations=1000):
    # estimate endpoints of arc on S^2 from unit-length points

    def _arc_distance(a, b):
        # arc distance between unit vectors a, b
        return np.arccos(a.dot(b)).item()

    def _is_on_segment(a, b, c):
        # test if c is on the arc segment formed by a and b
        # use inlier threshold as arbitrary choice of precision
        return abs(_arc_distance(a, c) + _arc_distance(b, c) - _arc_distance(a, b)) <= inlier_angle_rad_threshold

    def _distance_to_segment(a, b, c):
        # find arc distance from c to the arc formed by a and b
        a_x_b = np.cross(a, b)
        c_x_axb = np.cross(c, a_x_b)
        greater_circle_pt = np.cross(a_x_b, c_x_axb)
        if _is_on_segment(a, b, greater_circle_pt):
            return _arc_distance(greater_circle_pt, c)
        # if greater_circle_pt is not on the segment, return arc distance to the closest endpoint
        return min(
            _arc_distance(a, c),
            _arc_distance(b, c),
        )

    best_num_inliers = -1
    best_pt1 = None
    best_pt2 = None
    for i in range(iterations):
        print("Iteration " + str(i))
        rand_indices = np.random.choice(len(points), size=2, replace=False)
        pt1 = points[rand_indices[0]]
        pt2 = points[rand_indices[1]]

        best_pt1 = pt1
        best_pt2 = pt2

        num_inliers = 0
        for j in range(len(points)):
            pt = points[j]
            dist = _distance_to_segment(pt1, pt2, pt)
            if dist <= inlier_angle_rad_threshold:
                num_inliers += 1
        if num_inliers > best_num_inliers:
            print("Found new best num_inliers: " + str(num_inliers))
            best_num_inliers = num_inliers
            best_pt1 = pt1
            best_pt2 = pt2

    print("best num_inliers: " + str(best_num_inliers))

    return best_pt1, best_pt2
