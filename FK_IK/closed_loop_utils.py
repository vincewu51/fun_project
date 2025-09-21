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