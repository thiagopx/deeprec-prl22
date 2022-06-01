import numpy as np
import cv2

# import open3d as o3d
# from pycpd import RigidRegistration
from math import sin, cos, atan2, pi
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

# def run_cpd2d(points2d_src, points2d_tgt, invert=False, max_iterations=10):
#     ''' Point Set Registration: Coherent Point Drift - https://arxiv.org/pdf/0905.2635.pdf'''

#     reg = RigidRegistration(X=points2d_tgt, Y=points2d_src, max_iterations=max_iterations)
#     _, (s_reg, R_reg, t_reg) = reg.register()

#     transformation = np.eye(3)
#     print(R_reg.T, invert)
#     sR = s_reg * (R_reg.T if invert else R_reg)
#     transformation[: 2, : 2] = sR
#     transformation[: 2, 2] =  t_reg
#     print(transformation)
#     return transformation


# def run_icp2d(points2d_src, points2d_tgt, threshold=500):

#     # append 0 as depth
#     points3d_src = np.pad(points2d_src, ((0, 0), (0, 1)), 'constant', constant_values=0)
#     points3d_tgt = np.pad(points2d_tgt, ((0, 0), (0, 1)), 'constant', constant_values=0)

#     # numpy to point cloud
#     pcloud_src = o3d.geometry.PointCloud()
#     pcloud_tgt = o3d.geometry.PointCloud()
#     pcloud_src.points = o3d.utility.Vector3dVector(points3d_src)
#     pcloud_tgt.points = o3d.utility.Vector3dVector(points3d_tgt)

#     # run icp
#     reg_p2p = o3d.registration.registration_icp(pcloud_src, pcloud_tgt, threshold)#, criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

#     # 3d -> 2 matrix
#     transformation = np.eye(3)
#     # print(reg_p2p.transformation)
#     transformation[: 2, : 2] = reg_p2p.transformation[: 2, : 2] # rotation
#     transformation[: 2, 2] = reg_p2p.transformation[: 2, 3] # translation
#     assert reg_p2p.transformation[2, 2] >= 0.98
#     return transformation


from functools import partial


def kernel(threshold, error):
    if np.linalg.norm(error) < threshold:
        return 1.0
    return 0.0


def get_correspondence_indices(P, Q):

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(Q.T)
    P_idx = np.arange(P.shape[1]).reshape((-1, 1))
    distances, Q_chosen_idx = nbrs.kneighbors(P.T)
    return np.hstack([P_idx, Q_chosen_idx]).tolist()


# def get_correspondence_indices(P, Q):

#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(Q) # samples organized in rows
#     P_idx = np.arange(P.shape[1]).reshape((-1, 1))
#     distances, Q_chosen_idx = nbrs.kneighbors(P)
#     return np.hstack([P_idx, Q_chosen_idx]).tolist()


def dR(theta):
    return np.array([[-sin(theta), -cos(theta)], [cos(theta), -sin(theta)]])


def R(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


def jacobian(x, p_point):
    theta = x[2]
    J = np.zeros((2, 3))
    J[0:2, 0:2] = np.identity(2)
    J[0:2, [2]] = dR(0).dot(p_point)
    return J


def error(x, p_point, q_point):
    rotation = R(x[2])
    translation = x[:2]
    prediction = rotation.dot(p_point) + translation
    return prediction - q_point


def prepare_system(x, P, Q, correspondences, kernel=lambda distance: 1.0):
    H = np.zeros((3, 3))
    g = np.zeros((3, 1))
    chi = 0
    for i, j in correspondences:
        p_point = P[:, [i]]
        q_point = Q[:, [j]]
        e = error(x, p_point, q_point)
        weight = kernel(e)  # Please ignore this weight until you reach the end of the notebook.
        J = jacobian(x, p_point)
        H += weight * J.T.dot(J)
        g += weight * J.T.dot(e)
        chi += e.T * e
    return H, g, chi


def icp_least_squares(P, Q, iterations=30, kernel=lambda distance: 1.0):

    P = P.T
    Q = Q.T
    x = np.zeros((3, 1))
    chi_values = []
    P_copy = P.copy()
    # corresp_values = []
    W = np.eye(3)
    W_it = np.eye(3)
    for i in range(iterations):
        correspondences = get_correspondence_indices(P_copy, Q)
        # corresp_values.append(correspondences)
        H, g, chi = prepare_system(x, P, Q, correspondences, kernel)
        dx = np.linalg.lstsq(H, -g, rcond=None)[0]
        x += dx
        x[2] = atan2(sin(x[2]), cos(x[2]))  # normalize angle
        chi_values.append(chi.item(0))
        # x_values.append(x.copy())
        rot = R(x[2])
        t = x[:2]
        P_copy = rot.dot(P.copy()) + t
        W[:2, :2] = rot
        W[:2, [-1]] = t
    return W, chi_values  # , corresp_values


# ===========================================
# exhaustive search
# ===========================================

# def cost(points_src, points_tgt):

#     points_src = points_src.astype(np.int0)

#     points_tgt_dict = defaultdict(int)
#     for y, x in zip(points_tgt[:, 1], points_tgt[:, 0]):
#         points_tgt_dict[y] = max(points_tgt_dict[y], x)

#     points_src_dict = defaultdict(lambda: float('inf'))
#     for y, x in zip(points_src[:, 1], points_src[:, 0]):
#         points_src_dict[y] = min(points_src_dict[y], x)

#     costs = []
#     for y, x in points_tgt_dict.items():
#         costs.append((points_src_dict[y] - x) ** 2)
#     costs = np.array(costs)

#     points_tgt_dict[y]
#     # print(costs)

#     return costs[costs != np.inf].sum()


def best_rotation(points_src, points_tgt, max_theta=5.0, num_angles_per_degree=1, center=(0, 0)):
    """Try all rotations to find the best alignment. It assume points_src and
    points_tgt are roughly aligned.
    @max_theta: maximum angle range (in degrees).
    @num_angles: the number of angles (values) considered.
    """

    # x0, y0, _ = points_tgt.mean(axis=0).astype(np.int0)
    x0, y0, _ = center
    num_angles = int(2 * num_angles_per_degree * max_theta + 1)
    thetas = np.linspace(-max_theta, max_theta, num_angles)
    thetas_rad = thetas * np.pi / 180.0

    # defining rotation around center
    R = lambda theta: np.array(
        [[np.cos(theta), np.sin(theta), 0.0], [-np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    T = lambda x, y: np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])
    Rcenter = lambda theta: T(x0, y0) @ R(theta) @ T(-x0, -y0)

    ## calculate the best rotation
    #
    # cost function
    cost = lambda A, B: NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(B).kneighbors(A)[0].sum()
    costs = [cost(points_src @ Rcenter(theta).T, points_tgt) for theta in thetas_rad]
    # best angle
    best_theta = thetas_rad[np.argmin(costs)]
    return Rcenter(best_theta)


def best_translation(points_src, points_tgt, max_dx=5, max_dy=5):
    """Try all translation in a grid to find the best alignment. It assume points_src and
    points_tgt are aligned by the centroids.
    @max_dx: maximum horizontal shift.
    @max_dy: maximum vertical shift.
    """

    xx, yy = np.meshgrid(np.arange(-max_dx, max_dx + 1), np.arange(-max_dy, max_dy + 1))
    dxy = list(zip(xx.flatten(), yy.flatten()))

    # defining translation
    T = lambda x, y: np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])

    # cost function
    cost = lambda A, B: NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(B).kneighbors(A)[0].sum()

    costs = [cost(points_src @ T(dx, dy).T, points_tgt) for dx, dy in dxy]
    best_dx, best_dy = dxy[np.argmin(costs)]
    return T(best_dx, best_dy)


#     # corresp_values = []
#     W = np.eye(3)
#     W_it = np.eye(3)
#     for i in range(iterations):
#         correspondences = get_correspondence_indices(P_copy, Q)
#         # corresp_values.append(correspondences)
#         H, g, chi = prepare_system(x, P, Q, correspondences, kernel)
#         dx = np.linalg.lstsq(H, -g, rcond=None)[0]
#         x += dx
#         x[2] = atan2(sin(x[2]), cos(x[2])) # normalize angle
#         chi_values.append(chi.item(0))
#         # x_values.append(x.copy())
#         rot = R(x[2])
#         t = x[: 2]
#         P_copy = rot.dot(P.copy()) + t
#         W[: 2, : 2] = rot
#         W[: 2, [-1]] = t
#     return W, chi_values#, corresp_values

# for i in range(iterations):
#     # correspondences
#     distances, indices = nbrs.kneighbors(P_copy, return_distance=True) # remove 3-rd coordinate
#     distances = distances.ravel()
#     indices = indices.ravel()

#     # print(P.shape)
#     # print(Q[indices].shape)

#     H, g, chi = prepare_system(x, P_copy, Q[indices], kernel)
#     chi_values.append(chi.item(0))
#     dx = np.linalg.lstsq(H, -g, rcond=None)[0]
#     x += dx

#     # rigid transformation
#     x[2] = atan2(sin(x[2]), cos(x[2])) # normalize angle
#     rot = R(x[2])
#     t = x[: 2]
#     P_copy = P_copy @ rot + t.ravel()#  rot.dot(P.copy()) + t

# return W, chi_values


# def prepare_system(source_pts, target_pts):
#     num_points = source_pts.shape[0]

#     A = np.zeros((2 * num_points, 6)) # 6 dof (affine transformation)
#     b = np.zeros(2 * num_points)
#     i = 0
#     for src, tgt in zip(source_pts, target_pts):
#         # print(src, tgt)
#         A[i, : 3] = src
#         A[i + 1, 3 :] = src
#         b[i] = tgt[0]     # x
#         b[i + 1] = tgt[1] # y
#         i += 2
#     return A, b


# def icp_opencv(source_pts, target_pts, iterations=30, threshold=1.0, tolerance=0.001):

#     print(source_pts.shape[1])
#     assert source_pts.ndim == target_pts.ndim == 2
#     assert source_pts.shape[1] == target_pts.shape[1] == 3 # 2-D in homogeneous coordinates

#     # nearest neighbors: record source points
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
#     nbrs.fit(target_pts[:, : -1])

#     # auxiliary matrices
#     #
#     # final
#     W = np.eye(3)
#     # iteration
#     W_it = np.eye(3)

#     # iterative process
#     prev_error = 0
#     curr_pts = source_pts.copy()
#     for i in range(iterations):
#         # correspondences
#         distances, indices = nbrs.kneighbors(curr_pts[:, : -1], return_distance=True) # remove 3-rd coordinate
#         distances = distances.ravel()
#         indices = indices.ravel()

#         print((distances <= threshold).sum())
#         # apply distance threshold
#         indices = indices[distances <= threshold]
#         distances = distances[distances <= threshold]
#         print(indices.shape)

#         # calculate the affine transform
#         A, b = prepare_system(curr_pts, target_pts[indices])
#         x = np.linalg.lstsq(A, b, rcond=None)[0]

#         # update points and the final transf. matrix
#         W_it[: 2] = x.reshape((2, -1))  # affine parameters
#         curr_pts = curr_pts @ W_it.T
#         W = W_it @ W

#         # check error
#         mean_error = np.mean(distances)
#         if np.abs(prev_error - mean_error) < tolerance:
#             break
#         prev_error = mean_error
#     return W, mean_error
