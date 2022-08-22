#! /usr/bin/env python

import discretize
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG import (
    maps,
)
import matplotlib.pyplot as plt

nsoundings = 68

TIMES_HM_SKYTEM = np.logspace(-5,-2,100)

altitude = 40
radius = 10.415373695237049


cs, ncx, ncz, npad = 2, 50.0, 10.0, 28
hx = [(cs, ncx), (cs, npad, 1.25)]
ncp = 45
dp = 2 * np.pi / ncp  # cell width phi
hp =  [(dp, ncp)]
temp_belowsurface = np.logspace(np.log10(1), np.log10(24.0), 40)
temp_above_surface = temp_belowsurface[np.cumsum(temp_belowsurface) < altitude/2]
npad = 20
temp_pad = temp_belowsurface[-1] * 1.2 ** np.arange(npad)
temp_padabove = temp_above_surface[-1] * 1.4 **np.arange(npad)
hz = np.r_[temp_pad[::-1], temp_belowsurface[::-1], temp_above_surface,temp_above_surface[::-1],temp_above_surface, temp_padabove]

meshCyl = discretize.CylMesh([hx, hp, hz],)
meshCyl.x0 = [0,0, -np.sum(np.r_[temp_pad[::-1], temp_belowsurface[::-1],])]


print("the mesh has {} cells".format(meshCyl.nC))

##

x = np.logspace(-1, 1.4, 64)
x = x / np.sum(x) * 160

meshEC = discretize.TensorMesh([np.r_[50, np.ones(66)*20, 50], np.flip(x)], x0=[0, -160])
EC = np.flip(np.exp(np.loadtxt('data and simulations/synthetic_model_selection_quasi2D.txt')).reshape(68,-1), axis=1).reshape(-1,1,order='F')
w = meshEC.hx

# Mesh extendend verziltingskaart --> SEE 01-proof-of-concept/stap1_deliverable1_v3 (OcTree extension)
wl = 5000
wr = wl
depth_distance_below = 5000
depth_distance_above = 0

mesh_verziltingskaart_extended = discretize.TensorMesh(
    [
        np.r_[wl, w[1:-1], wr],
        np.r_[depth_distance_below, meshEC.hy[:-1]],
    ], )

LM_waveform = pd.read_excel('data and simulations/LM_waveform.xls')
t0_lm = min(LM_waveform['Time'].values)
tmax_lm_offtime =  max(LM_waveform['Time'].values)

tussenmesh = discretize.TensorMesh(
    [
        np.r_[wl, w[1:-1], wr],
        np.r_[meshCyl.hz],
    ], x0=[0, meshCyl.x0[2]] )

mtest = np.ones(tussenmesh.shape_cells) * np.flip(np.flip(EC)[0:68])
# mtest = np.ones(tussenmesh.shape_cells) * np.flip(np.flip(EC)[0]) # 1D-isation
for idx, i in enumerate(-np.cumsum(x)):
    print(i)
    mtest[:,tussenmesh.vectorCCy < i] = np.flip(np.flip(EC)[68*idx:68*(idx+1)])
    # mtest[:, tussenmesh.vectorCCy < i] = np.flip(EC)[68 * idx] # 1D-isation

mtest[:,tussenmesh.vectorCCy >= 0] = 1e-8



HM_waveform = pd.read_excel('data and simulations/HM_waveform.xls')
t0_hm = min(HM_waveform['Time'].values)
tmax_hm_offtime =  max(HM_waveform['Time'].values)


##
def wave_function_hm(t):
    x = np.r_[t0_hm, HM_waveform['Time'].values, 1e-1]
    y = np.r_[0, HM_waveform['Current'].values, 0]
    f = interp1d(x, y)
    return f(t)

waveform_hm = tdem.sources.RawWaveform(waveFct=wave_function_hm, offTime=0.)

##
for sounding in np.arange(0, nsoundings):

    src_z = altitude
    rx_z = src_z + 2.0

    tussenmesh = discretize.TensorMesh(
        [
            np.r_[wl, w[1:-1], wr],
            np.r_[meshCyl.hz],
        ], x0=[-np.sum(np.r_[wl, w[1:sounding], w[sounding] / 2,]),
               meshCyl.x0[2]])

    srcList = []


    rx = tdem.Rx.PointMagneticFluxTimeDerivative(
        locations=np.r_[13.2, 0, rx_z], times=TIMES_HM_SKYTEM, orientation='z',
    )

    src = tdem.Src.CircularLoop(
        receiver_list=[rx],
        loc=np.r_[0.0, 0.0, src_z],  # average height as stated in DATA REPORT,
        orientation='z',
        radius=radius,
        N=1,  # number of turns
        # srcType='inductive',
        # waveform=waveform_hm
    )

    srcList.append(src)

    survey = tdem.Survey(srcList)

    plotmesh_twosides = discretize.TensorMesh(
        [
            np.r_[np.flip(meshCyl.hx), meshCyl.hx], meshCyl.hz
        ], x0=[-np.sum(meshCyl.hx), meshCyl.x0[2]]
    )
    mapping = maps.Mesh2Mesh([plotmesh_twosides, tussenmesh])
    m_bothsides = mapping * (mtest.flatten(order='F'))

    M_final = np.zeros(meshCyl.n_cells)
    for r in meshCyl.vectorCCx:
        for theta in meshCyl.vectorCCy:
            minarg = np.argmin(np.abs(r * np.cos(theta) - plotmesh_twosides.cell_centers[:, 0]))
            minval = plotmesh_twosides.cell_centers[minarg, 0]
            M_final[np.all([meshCyl.cell_centers[:, 0] == r, meshCyl.cell_centers[:, 1] == theta, ], axis=0)] = \
                m_bothsides[plotmesh_twosides.cell_centers[:, 0] == minval]

    time_steps = [
            # (np.abs(t0_hm) / 20, 20), # peaktime
            # (np.abs(tmax_hm_offtime) / 20, 20),  # offtime
            (3e-7, 100),
            (1e-6, 200),
            (3e-6, 200),
            (1e-5, 100),
            (3e-5, 100),
            (1e-4, 100),

    ]
    prob = tdem.Simulation3DMagneticFluxDensity(
        meshCyl, survey=survey, sigmaMap=maps.IdentityMap(meshCyl), time_steps=time_steps,# t0=t0_hm
    )

    fields = prob.fields(M_final)
    dclean = prob.dpred(M_final, f=fields)
    np.savetxt('synthetic_PERFECT_dclean_2D_hm_sounding'+str(sounding)+'.txt', dclean)

##

