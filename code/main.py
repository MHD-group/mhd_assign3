#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Created On  : 2023-04-03 00:24
# Last Modified : 2023-04-03 16:58
# Copyright © 2023 myron <yh131996@mail.ustc.edu.cn>
#
# Distributed under terms of the MIT license.


import numpy as np
from numpy import arange, cos, sqrt, sin, abs
from numpy import pi as π
from matplotlib import pyplot as plt
import argparse

# wave's shape1
def func1(x, C=1, t=0):
    conds = [x < -0.4,\
             np.logical_and(x < -0.2, x >= -0.4),\
             np.logical_and(x >= -0.2, x < -0.1),\
             np.logical_and(x >= -0.1, x < 0),\
             x >= 0]
    funcs = [0,\
             lambda x:1.0-abs(x+0.3)/0.1,\
             0,\
             1,\
             0]
    return np.roll(np.piecewise(x, conds, funcs), int(t*C))

# wave's shape2
def func2(x, C=1, t=0):
    conds = [x < -0.8,\
             np.logical_and(x < -0.3, x >= -0.8),\
             np.logical_and(x < 0, x >= -0.3),\
             x >= 0]
    funcs = [1.8,\
             lambda x : 1.4 + 0.4 * cos(2*π * (x+0.8) ),\
             1.0,\
             1.8]
    return np.roll(np.piecewise(x, conds, funcs), int(t*C))

def Upwind(x, C=1, t=1):
    N = x.size
    tmp = np.zeros((2, N), dtype=x.dtype)
    tmp[0] = x.copy()
    tmp[1] = x.copy()
    result = tmp[0]
    for n in range(t):
        cur = n%2
        nex = (n%2 + 1)%2
        for i in range(N-1):
            I = i - 1
            tmp[nex,I+1] =  tmp[cur,I+1] - C*(tmp[cur, I+1] - tmp[cur, I])
            result = tmp[nex]
    return tmp[0]

def minmod(a, b):
    return 0 if  a * b < 0 else min([a, b]) if b > 0 else max([a, b])

def limiter(x, C=1, t=1):
    N = x.size
    tmp = np.zeros((2, N), dtype=x.dtype)
    tmp[0] = x.copy()
    tmp[1] = x.copy()
    result = tmp[0]
    for n in range(t):
        cur = n%2
        nex = (n%2 + 1)%2
        for i in range(N):
            I = i - 2
            tmp[nex,I+1] =  tmp[cur,I+1] - C*(tmp[cur, I+1] - tmp[cur, I]) - 0.5 * C * (1 - C) *\
			( minmod(tmp[cur, I+1]-tmp[cur, I], tmp[cur, I+2]-tmp[cur, I+1]) - \
                        minmod(tmp[cur, I]-tmp[cur, I-1], tmp[cur, I+1]-tmp[cur, I]) )
            result = tmp[nex]

    return result

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculate X to the power of Y")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("-x", "--resolution", default=0.01, type=float, help="length of Δx")
    parser.add_argument("-C", "--ratio", default=0.5, type=float, help="Δt/Δx")
    parser.add_argument("-i", "--input", default=1, type=int, help="f(x) when t=0")
    parser.add_argument("-m", "--method", default="limiter", type=str, help="methods")
    args = parser.parse_args()

    # Δx: args.resolution
    x = np.arange(-1, 2, args.resolution)
    # C is Δt/Δx
    C = args.ratio
    # Δt
    t = C * args.resolution

    T = 0.5
    n_t = int(T/t)

    print(t, n_t)
    # math output
    if args.input == 1:
        M0 = func1(x, C)
        M1 = func1(x, C, n_t)
    elif args.input == 2:
        M0 = func2(x, C)
        M1 = func2(x, C, n_t)
    else:
        print("error input function")

    # simu output
    if args.method == "Upwind":
        S1 = Upwind(M0, C, n_t)
    elif args.method == "limiter":
        S1 = limiter(M0, C, n_t)
    else:
        print("error input function")
    fig, axs = plt.subplots(2,
                            1,
                            figsize=(8, 6))
    axs[0].plot(x, M0)
    axs[1].plot(x, M1, alpha=0.5)
    axs[0].set_xlim(-0.5,1)
    axs[1].set_xlim(-0.5,1)
    axs[1].plot(x, S1, '--o')
    plt.show()


