import numpy as np
from scipy.optimize import fsolve

def four_bar(theta1, L1, L2, L3, L4):
    """
    Solve 4-bar linkage closure equations.
    Ground link: L1, input crank: L2, coupler: L3, output rocker: L4.
    
    theta1: input angle (radians)
    Returns: theta2, theta3 (angles of coupler and rocker)
    """
    def equations(vars):
        theta2, theta3 = vars
        eq1 = L2*np.cos(theta1) + L3*np.cos(theta2) - L1 - L4*np.cos(theta3)
        eq2 = L2*np.sin(theta1) + L3*np.sin(theta2) - L4*np.sin(theta3)
        return [eq1, eq2]
    
    sol = fsolve(equations, [0.1, 0.1])
    return sol

def two_arm_parallel(xd, yd, L):
    """
    Two identical arms anchored at (0,0) and (d,0) grasping the same rigid bar endpoint.
    Each arm: 2-link planar manipulator with equal link length L.
    Goal: find joint angles satisfying closed-loop grasp.
    """
    d = 2*L
    
    def equations(vars):
        th1, th2, th3, th4 = vars
        x1 = L*np.cos(th1) + L*np.cos(th1+th2)
        y1 = L*np.sin(th1) + L*np.sin(th1+th2)
        x2 = d + L*np.cos(th3) + L*np.cos(th3+th4)
        y2 = L*np.sin(th3) + L*np.sin(th3+th4)
        return [x1-xd, y1-yd, x2-xd, y2-yd]
    
    sol = fsolve(equations, [0.5, 0.5, -0.5, -0.5])
    return sol

def stewart_platform(base_pts, plat_pts, pose):
    """Leg lengths of Stewart–Gough platform given base, platform points and pose [x,y,z,roll,pitch,yaw]."""
    x,y,z,roll,pitch,yaw = pose
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    R = Rz @ Ry @ Rx
    t = np.array([x,y,z])
    lengths = []
    for Bi, Pi in zip(base_pts, plat_pts):
        Pi_world = R @ Pi + t
        lengths.append(np.linalg.norm(Pi_world - Bi))
    return np.array(lengths)

def rpr_3dof(base_pts, plat_pts, pose):
    """Leg lengths for 3×RPR planar parallel manipulator."""
    x,y,phi = pose
    R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    lengths = []
    for Bi, Pi in zip(base_pts, plat_pts):
        Pi_world = R @ Pi + np.array([x,y])
        lengths.append(np.linalg.norm(Pi_world - Bi))
    return np.array(lengths)

def jacobian_numeric(f, q, eps=1e-6):
    """Numerical Jacobian of function f at q."""
    q = np.array(q, dtype=float)
    m = len(f(q))
    n = len(q)
    J = np.zeros((m,n))
    for i in range(n):
        dq = np.zeros_like(q)
        dq[i] = eps
        J[:,i] = (f(q+dq)-f(q-dq))/(2*eps)
    return J

def is_singular(J, tol=1e-6):
    """Check if Jacobian is singular."""
    return np.linalg.matrix_rank(J, tol) < min(J.shape)
