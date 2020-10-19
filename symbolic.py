#!/bin/env python3
import numpy as np
import sympy as sym

theta = sym.Symbol('theta')
dt =    sym.Symbol('dt')
sigma_a =    sym.Symbol('sigma_a')
sigma_w =    sym.Symbol('sigma_w')

distMatrix = np.diag([sigma_a**2, sigma_a**2, sigma_w**2])
c = sym.cos(theta)
s = sym.sin(theta)
G = np.array([
        [0      ,0      ,0],
        [0      ,0      ,0],
        [dt*c   ,dt*s   ,0],
        [-dt*s  ,dt*c   ,0],
        [0      ,0      ,dt],
    ])
Q = G @ distMatrix @ np.transpose(G)
sym.pprint(sym.Matrix(Q))
