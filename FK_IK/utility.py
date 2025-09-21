import numpy as np

def skew_sym(v):
    """Skew-symmetric matrix of a 3D vector v."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def MatrixExp6(Sigma):
    """Matrix exponential for a screw motion Sigma."""
    omega = Sigma[:3]  # Angular part of screw axis
    v = Sigma[3:]      # Linear part of screw axis
    theta = np.linalg.norm(omega)
    
    if theta == 0:
        R = np.eye(3)
        V = np.eye(3)
    else:
        R = np.eye(3) + (np.sin(theta) / theta) * skew_sym(omega) + ((1 - np.cos(theta)) / (theta ** 2)) * np.dot(skew_sym(omega), skew_sym(omega))
        V = np.eye(3) + (1 - np.cos(theta)) / (theta ** 2) * skew_sym(omega) + (theta - np.sin(theta)) / (theta ** 3) * np.dot(skew_sym(omega), skew_sym(omega))
    
    return np.vstack([np.hstack([R, np.dot(V, v.reshape(3, 1))]), [0, 0, 0, 1]])

def forward_kinematics_space(M, Slist, thetalist):
    """Forward kinematics calculation."""
    T = np.eye(4)
    for i in range(len(thetalist)):
        T = np.dot(T, MatrixExp6(Slist[:, i] * thetalist[i]))
    return np.dot(T, M)

def jacobian(Slist, thetalist, M):
    """Compute the Jacobian matrix using forward kinematics."""
    J = np.zeros((6, len(thetalist)))
    T = np.eye(4)
    
    for i in range(len(thetalist)):
        T = np.dot(T, MatrixExp6(Slist[:, i] * thetalist[i]))
        
        # The screw axis at joint i
        screw = Slist[:, i]
        
        # Compute the position vector from the base to the end-effector
        position = T[:3, 3] - M[:3, 3]
        
        # Angular and linear components of the screw axis
        omega = screw[:3]  # Angular part
        v = screw[3:]      # Linear part
        
        # Apply the inverse rotation matrix to the angular part
        omega_rot = np.dot(np.linalg.inv(T[:3, :3]), omega)
        
        # The Jacobian consists of the angular part (omega_rot) and linear part (v)
        J[:, i] = np.hstack([omega_rot, v])
    
    return J

def inverse_kinematics_space(Slist, M, T_sd, thetalist0, eomg=1e-3, ev=1e-3, max_iterations=100):
    thetalist = thetalist0
    for _ in range(max_iterations):
        T_current = forward_kinematics_space(M, Slist, thetalist)
        
        # Compute the position and orientation errors
        position_error = T_current[:3, 3] - T_sd[:3, 3]
        orientation_error = rotation_error(T_current, T_sd)
        
        # Combine the errors
        error = np.hstack([position_error, orientation_error])
        
        # If error is small enough, stop
        if np.linalg.norm(error) < ev:
            return thetalist, True
        
        # Compute the Jacobian and update thetas
        J = jacobian(Slist, thetalist, M)
        dtheta = np.dot(np.linalg.pinv(J), error)
        thetalist += dtheta
    
    return thetalist, False

def rotation_error(T_current, T_sd):
    """Compute the rotation error between two transformations."""
    R_current = T_current[:3, :3]
    R_sd = T_sd[:3, :3]
    
    # Compute the rotation matrix error (R_error)
    R_error = np.dot(R_current.T, R_sd)
    
    # Compute axis and angle using Rodrigues' formula
    theta = np.arccos((np.trace(R_error) - 1) / 2)
    
    if theta == 0:
        return np.zeros(3)  # No rotation error if angle is 0
    
    # Compute the axis of rotation
    axis = (R_error[2, 1] - R_error[1, 2], 
            R_error[0, 2] - R_error[2, 0], 
            R_error[1, 0] - R_error[0, 1])
    
    axis = np.array(axis)  # Convert the axis to a numpy array
    return axis * theta / np.linalg.norm(axis)  # Return the scaled rotation error