#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Created On  : 2023-04-03 00:24
# Last Modified : 2023-04-26 03:27
# Copyright © 2023 myron <yh131996@mail.ustc.edu.cn>
#
# Distributed under terms of the MIT license.


import numpy as np
from numpy import arange, cos, sqrt, sin, abs
from numpy import pi as π
from matplotlib import pyplot as plt
import argparse

# wave's init shape
def init_w(x):
    N = x.size
    tmp = np.zeros((3, N), dtype=x.dtype)
    conds = [x < -0, x >= 0]
    funcs = [[0.445, 0.5], [0.311, 0], [8.928, 1.4275]]
    for i in range(3):
        tmp[i] = np.piecewise(x, conds, funcs[i])
    #print(tmp.shape)
    #print(tmp[:,0:1].shape)
    #print("----------")
    tmp1 = tmp.transpose(1, 0)[:,:,np.newaxis]
    #print(tmp1.shape)
    #print(tmp1[:,1,0].shape)
#    print(tmp1[0:1,:,np.newaxis].shape)
    return tmp.transpose(1,0)[:,:,np.newaxis]

# conservation
def w2U(w, γ=1.4):
    u=w.copy()
    u[:, 0, 0] = w[:, 0, 0]
    u[:, 1, 0] = w[:, 1, 0] / u[:, 0, 0]
    u[:, 2, 0] = (γ-1) * (w[:,2,0] - 0.5 * u[:,0,0] * u[:,1,0]**2)
    return u

# vector to Matrix
def w2A(w, γ=1.4):
    U = w2U(w, γ)
    ρ = w[:,0,0]
    m = w[:,1,0]
    u = U[:,1,0]
    E = w[:,2,0]
    A = np.zeros((ρ.size, 3, 3), float)
    A[:,0,1] = 1
    A[:,1,0] = 0.5*(u**2)*(γ-3)
    A[:,1,1] = -u*(γ-3)
    A[:,1,2] = (γ-1)
    A[:,2,0] = (γ-1)*(u**3)-γ*u*E/ρ
    A[:,2,1] = (γ/ρ)*E - 1.5*(γ-1)*u**2
    A[:,2,2] = γ*u
    return A

def w2F(w, γ=1.4):
    U = w2U(w, γ)
    u = U[:,1:2,:]
    p = U[:,2:3,:]
    sub = np.concatenate((np.zeros(p.shape, float),\
            p, p*u), axis=1)
    F = u*w+sub
    return F

def w2R(w, γ=1.4):
    U = w2U(w, γ)
    ρ = w[:,0,0]
    m = w[:,1,0]
    E = w[:,2,0]
    u = U[:,1,0]
    p = U[:,2,0]
    a = sqrt(γ*p/ρ)
    H = (a**2)/(γ-1) + 0.5*u**2

    R = np.zeros((ρ.size, 3, 3), float)
    R[:,0,0] = 1
    R[:,0,1] = 1
    R[:,0,2] = 1
    R[:,1,0] = u-a
    R[:,1,1] = u
    R[:,1,2] = u+a
    R[:,2,0] = H-u*a
    R[:,2,1] = 0.5*u**2
    R[:,2,2] = H+u*a
    return R

# vector to Matrix
def w2L(w, γ=1.4):
    U = w2U(w, γ)
    ρ = w[:,0,0]
    m = w[:,1,0]
    E = w[:,2,0]
    u = U[:,1,0]
    p = U[:,2,0]
    a = sqrt(γ*p/ρ)
    H = (a**2)/(γ-1) + 0.5*u**2
    K = 0.5*(γ-1/a**2)

    L = np.zeros((ρ.size, 3, 3), float)
    L[:,0,0] = K * 0.5*u*(u+2*a/(γ-1))
    L[:,0,1] = K * -(u+a/(γ-1))
    L[:,0,2] = K * 1
    L[:,1,0] = K * 2*(H-u**2)
    L[:,1,1] = K * 2*u
    L[:,1,2] = K * -2
    L[:,2,0] = K * 0.5*u*(u-2*a/(γ-1))
    L[:,2,1] = K * -(u-a/(γ-1))
    L[:,2,2] = K * 1
    return L

# vector to Matrix
def U2λ(U, γ=1.4):
    ρ = U[:,0,0]
    u = U[:,1,0]
    p = U[:,2,0]
    a = sqrt(γ*p/ρ)
    λ = np.zeros((ρ.size, 3, 3), float)
    λ[:,0,0] = u-a
    λ[:,1,1] = u
    λ[:,2,2] = u+a
    return λ

def w2λ(w, γ=1.4):
    return U2λ(w2U(w, γ), γ)


# non conservation
# vector to Matrix
def U2R(U, γ=1.4):
    ρ = U[:,0,0]
    u = U[:,1,0]
    p = U[:,2,0]
    R = np.zeros((ρ.size, 3, 3), float)
    a = sqrt(γ*p/ρ)
    R[:,0,0] = 0.5/(a**2)
    R[:,0,1] = 1/(a**2)
    R[:,0,2] = 0.5/(a**2)
    R[:,1,0] = -0.5/(ρ*a)
    R[:,1,1] = 0
    R[:,1,2] = 0.5/(ρ*a)
    R[:,2,0] = 0.5
    R[:,2,1] = 0
    R[:,2,2] = 0.5
    return R

# vector to Matrix
def U2L(U, γ=1.4):
    ρ = U[:,0,0]
    u = U[:,1,0]
    p = U[:,2,0]
    L = np.zeros((ρ.size, 3, 3), float)
    a = sqrt(γ*p/ρ)
    L[:,0,0] = 0
    L[:,0,1] = -ρ*a
    L[:,0,2] = 1
    L[:,1,0] = a**2
    L[:,1,1] = 0
    L[:,1,2] = -1
    L[:,2,0] = 0
    L[:,2,1] = ρ*a
    L[:,2,2] = 1
    return L

def func(w, γ = 1.4):
    ρ = w[0]
    m = w[1]
    E = w[2]
    u = m/ρ
    res = w.copy()
    res[0] = m
    res[1] = (γ - 1)*E + (3 - γ)*m*m/(2*ρ)
    res[2] = (γ*E - (γ - 1)*m*m/(2*ρ))*m/ρ
    return res


# wave's ref shape from excel
def funcRef(x, C=1, t=0):
    conds = [x < -0.8,\
             np.logical_and(x < -0.3, x >= -0.8),\
             np.logical_and(x < 0, x >= -0.3),\
             x >= 0]
    funcs = [1.8,\
             lambda x : 1.4 + 0.4 * cos(2*π * (x+0.8) ),\
             1.0,\
             1.8]
    return np.roll(np.piecewise(x, conds, funcs), int(t*C))

def Upwind(w, fw, γ=1.4, C=0.5, t=100):
    N = w[1,:].size
    print(w.shape, w.size)
    tmp_w = np.expand_dims(w, 0).repeat(2, axis=0)
    print(tmp_w.shape, tmp_w.size)
    print(N)
    tmp_w[0] = w.copy()
    tmp_w[1] = w.copy()
    tmp_fw = np.expand_dims(fw, 0).repeat(2, axis=0)
    tmp_fw[0] = fw.copy()
    tmp_fw[1] = fw.copy()
    for n in range(5):
        cur = n%2
        nex = (n%2 + 1)%2
        print("!")
        for i in range(N):
            I = i - 1
            tmp_w[nex, 0, I+1] =  tmp_w[cur, 0, I+1] - C*(tmp_fw[cur, 0, I+1] - tmp_fw[cur, 0, I])
            tmp_w[nex, 1, I+1] =  tmp_w[cur, 1, I+1] - C*(tmp_fw[cur, 1, I+1] - tmp_fw[cur, 1, I])
            tmp_w[nex, 2, I+1] =  tmp_w[cur, 2, I+1] - C*(tmp_fw[cur, 2, I+1] - tmp_fw[cur, 2, I])
            result = tmp_w[nex]
            tmp_fw[nex] = func(tmp_w[nex], γ)
    return result

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
    parser.add_argument("-m", "--methods", default="Upwind,Minmod", type=str, help="methods")
    parser.add_argument("-t", "--times", default="0.5,0.75", type=str, help="time")
    args = parser.parse_args()

    Ts = [float(idx) for idx in args.times.split(',')]
    methods = args.methods.split(',')
    print(Ts)
    # Δx: args.resolution
    x = np.arange(-1, 1, args.resolution)
    # C is Δt/Δx
    C = args.ratio
    # Δt
    t = C * args.resolution
    T = Ts[0]

    method = methods[0]
    w = np.array([[[1],[2],[3]],[[2],[3],[4]]], dtype=float)
    print('w:', w)
    u = w2U(w)
    print('u:', u)
    A = w2A(w)
    print('A:',A)
    F = w2F(w)
    print('F:',F)
    L = U2L(u)
    print('L:',L)
    R = U2R(u)
    print('R:',R)
    L = w2L(w)
    print('L:',L)
    R = w2R(w)
    print('R:',R)
    λ = U2λ(u)
    print('λ:',λ)
    λ = w2λ(w)
    print('λ:',λ)

#    print(T, method)
#
#    n_t = int(T/t)
#
#    γ = 1.4
#
#    print(t, n_t)
#    # get input
#    if args.input == 1:
#        M0 = init_w(x)
#    elif args.input == 2:
#        M1 = init_w(x)
#    else:
#        print("error input function")
#    M2 = func(M0)
#
#    # simu output
#    print("γ = " + str(γ) + ", C = " + str(C) + ", n_t = "+ str(n_t))
#    if method == "Upwind":
#        print(0)
#        S1 = Upwind(M0, M2, γ, C, n_t)
#        print(1)
#    else:
#        print("error input function")
#    fig, axs = plt.subplots(3,
#                            3,
#                            figsize=(8, 6))
#    axs[0][0].plot(x, M0[0])
#    axs[1][0].plot(x, M0[1])
#    axs[2][0].plot(x, M0[2])
##    axs[0][0].plot(x, M1[0])
##    axs[1][0].plot(x, M1[1])
##    axs[2][0].plot(x, M1[2])
#    axs[0][1].plot(x, M2[0])
#    axs[1][1].plot(x, M2[1])
#    axs[2][1].plot(x, M2[2])
#    axs[0][2].plot(x, S1[0])
#    axs[1][2].plot(x, S1[1])
#    axs[2][2].plot(x, S1[2])
#    plt.show()


