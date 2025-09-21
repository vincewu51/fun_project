import numpy as np

def skew_sym(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def MatrixExp6(Sigma):
    theta = np.linalg.norm(Sigma[:3, 3])
    if theta == 0:
        return np.eye(4)
    R = np.eye(3) + (np.sin(theta) / theta) * skew_sym(Sigma[:3, 3]) + ((1 - np.cos(theta)) / (theta ** 2)) * np.dot(skew_sym(Sigma[:3, 3]), skew_sym(Sigma[:3, 3]))
    V = np.eye(3) + (1 - np.cos(theta)) / (theta ** 2) * skew_sym(Sigma[:3, 3]) + (theta - np.sin(theta)) / (theta ** 3) * np.dot(skew_sym(Sigma[:3, 3]), skew_sym(Sigma[:3, 3]))
    return np.vstack([np.hstack([R, np.dot(V, Sigma[:3, 3].reshape(3, 1))]), [0, 0, 0, 1]])

def forward_kinematics_space(M, Slist, thetalist):
    T = np.eye(4)
    for i in range(len(thetalist)):
        T = np.dot(T, MatrixExp6(Slist[:, i] * thetalist[i]))
    return np.dot(T, M)

def skew_sym_6(v):
    return np.array([[0, -v[2], v[1], v[3]],
                     [v[2], 0, -v[0], v[4]],
                     [-v[1], v[0], 0, v[5]],
                     [0, 0, 0, 0]])

def jacobian(Slist, thetalist):
    J = np.zeros((6, len(thetalist)))
    T = np.eye(4)
    for i in range(len(thetalist)):
        T = np.dot(T, MatrixExp6(Slist[:, i] * thetalist[i]))
        J[:, i] = np.dot(np.linalg.inv(T[:3, :3]), np.hstack([np.cross(Slist[:3, i], (T[:3, 3] - M[:3, 3])), Slist[:3, i]]))
    return J

def inverse_kinematics_space(Slist, M, T_sd, thetalist0, eomg=1e-3, ev=1e-3, max_iterations=100):
    thetalist = thetalist0
    for _ in range(max_iterations):
        T_current = forward_kinematics_space(M, Slist, thetalist)
        error = np.hstack([np.array([T_current[0, 3], T_current[1, 3], T_current[2, 3]]) - T_sd[:3, 3],
                           np.array([T_current[0, 2], T_current[1, 2], T_current[2, 2]]) - T_sd[:3, 2]])
        if np.linalg.norm(error) < ev:
            return thetalist, True
        J = jacobian(Slist, thetalist)
        dtheta = np.dot(np.linalg.pinv(J), error)
        thetalist += dtheta
    return thetalist, False