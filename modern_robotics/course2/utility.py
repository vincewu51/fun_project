import numpy as np
import modern_robotics as mr
import csv

def IKinBodyIterates(Blist, M, T_sd, thetalist0, eomg=1e-3, ev=1e-4, filename="iterates.csv"):
    """
    Extended Inverse Kinematics in Body Frame with iteration reporting.

    Arguments:
        Blist: 6xn screw axes in body frame
        M: Home configuration (4x4 SE3)
        T_sd: Desired configuration (4x4 SE3)
        thetalist0: Initial guess (n-vector)
        eomg: Angular error tolerance
        ev: Linear error tolerance
        filename: CSV file to save the iterates
    """

    thetalist = np.array(thetalist0).astype(float).copy()
    joint_history = [thetalist.copy()]

    i = 0
    max_iter = 20

    while i < max_iter:
        # Current end-effector configuration
        T_sb = mr.FKinBody(M, Blist, thetalist)

        # Error twist
        T_bd = np.dot(mr.TransInv(T_sb), T_sd)
        Vb = mr.se3ToVec(mr.MatrixLog6(T_bd))

        omega_b = Vb[0:3]
        v_b = Vb[3:6]
        err_omega = np.linalg.norm(omega_b)
        err_v = np.linalg.norm(v_b)

        # --- Print iteration report ---
        print(f"\nIteration {i}:")
        print("joint vector:", ", ".join([f"{x:.3f}" for x in thetalist]))
        print("SE(3) end-effector config:")
        print(np.array_str(T_sb, precision=3, suppress_small=True))
        print("error twist V_b:", ", ".join([f"{x:.3f}" for x in Vb]))
        print(f"angular error magnitude ||omega_b||: {err_omega:.3f}")
        print(f"linear error magnitude ||v_b||: {err_v:.3f}")

        # Check convergence
        if err_omega < eomg and err_v < ev:
            break

        # Compute body Jacobian and update
        Jb = mr.JacobianBody(Blist, thetalist)
        thetalist = thetalist + np.dot(np.linalg.pinv(Jb), Vb)

        # Save new iterate
        joint_history.append(thetalist.copy())
        i += 1

    # Save matrix to CSV
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in joint_history:
            writer.writerow([f"{x:.6f}" for x in row])

    return np.array(joint_history)
