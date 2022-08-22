#! /usr/bin/env python

import discretize
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG import (
    maps,
)

start = 0
nsoundings = 68
version = 'model_selection'

TIMES_LM_SKYTEM = np.r_[
    1.033999999999999926e-05, 1.285000000000000064e-05, 1.633999999999999941e-05, 2.084000000000000037e-05,
    2.633999999999999853e-05, 3.334999999999999748e-05, 4.233999999999999781e-05, 5.383999999999999951e-05,
    6.833999999999999960e-05, 8.634000000000000344e-05, 1.088000000000000019e-04, 1.428000000000000016e-04,
    1.857999999999999897e-04, 2.288000000000000049e-04, 2.717999999999999930e-04, 3.362999999999999887e-04,
    4.223000000000000191e-04, 5.297999999999999758e-04,]

altitude = 40
radius = 10.415373695237049


cs, ncx, ncz, npad = 5, 10.0, 10.0, 15
hx = [(cs, ncx), (cs, npad, 1.2)]
ncp = 22
dp = 2 * np.pi / ncp  # cell width phi
hp =  [(dp, ncp)]

npad = 12
temp_belowsurface = np.logspace(np.log10(1), np.log10(12.0), 30)
temp_above_surface = temp_belowsurface[np.cumsum(temp_belowsurface) < altitude/2]

temp_pad = temp_belowsurface[-1] * 1.2 ** np.arange(npad)
temp_padabove = temp_above_surface[-1] * 1.35 **np.arange(npad)
hz = np.r_[temp_pad[::-1], temp_belowsurface[::-1], temp_above_surface,temp_above_surface[::-1],temp_above_surface, temp_padabove]

meshCyl = discretize.CylMesh([hx,hp , hz],)
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


##
def wave_function_lm(t):
    x = np.r_[t0_lm, LM_waveform['Time'].values, 1e-1]
    y = np.r_[0, LM_waveform['Current'].values, 0]
    f = interp1d(x, y)
    return f(t)

waveform_lm = tdem.sources.RawWaveform(waveFct=wave_function_lm, offTime=0.)

##
for sounding in np.arange(start, start + 68):
    text = '_' + str(sounding) + '_' + version

    src_z = altitude
    rx_z = src_z + 2.0

    mesh_verziltingskaart_extended = discretize.TensorMesh(
        [
            np.r_[wl, w[1:-1], wr],
            np.r_[depth_distance_below, meshEC.hy[:-1],],
        ], x0=[-np.sum(np.r_[wl, w[1:sounding], w[sounding] / 2,]),
               -np.sum(mesh_verziltingskaart_extended.h[1])])

    # Mesh 2D projection of forward mesh without active cells
    mesh_fw_2D = discretize.TensorMesh(
        [
            np.r_[np.flip(meshCyl.hx), meshCyl.hx],
            meshCyl.hz[meshCyl.cell_centers_z < 0],
        ], )
    mesh_fw_2D.origin = np.r_[- np.sum(meshCyl.hx), meshCyl.x0[2]]

    # Mesh 2D projection of forward mesh with active cells
    mesh_fw_2D_active = discretize.TensorMesh(
        [
            mesh_fw_2D.hx,
            meshCyl.hz,
        ], )
    mesh_fw_2D_active.origin = mesh_fw_2D.x0

    mv2mfw = maps.Mesh2Mesh([mesh_fw_2D, mesh_verziltingskaart_extended])
    active = mesh_fw_2D_active.gridCC[:, 1] < 0.0
    mact = maps.InjectActiveCells(mesh_fw_2D_active, active, 1e-8)
    # m2to3 = maps.Surject2Dto3D(mesh)

    m = np.squeeze(EC.reshape(-1, 1, order='F'))


    srcList = []


    rx = tdem.Rx.PointMagneticFluxTimeDerivative(
        locations=np.r_[13.2, 0, rx_z], times=TIMES_LM_SKYTEM, orientation='z',
    )

    src = tdem.Src.CircularLoop(
        receiver_list=[rx],
        loc=np.r_[0.0, 0.0, src_z],  # average height as stated in DATA REPORT,
        orientation='z',
        radius=radius,
        N=1,  # number of turns
        srcType='inductive',
        waveform=waveform_lm
    )

    srcList.append(src)

    survey = tdem.Survey(srcList)

    plotmesh = discretize.TensorMesh(
        [
            meshCyl.hx, meshCyl.hz
        ]

    )

    plotmesh_twosides = discretize.TensorMesh(
        [
            np.r_[np.flip(meshCyl.hx), meshCyl.hx], meshCyl.hz
        ], x0=[-np.sum(meshCyl.hx), meshCyl.x0[2]]
    )
    m_bothsides = mact * mv2mfw * m


    M_final = np.zeros(meshCyl.n_cells)
    for r in meshCyl.vectorCCx:
        for theta in meshCyl.vectorCCy:
            minarg = np.argmin(np.abs(r * np.cos(theta) - plotmesh_twosides.cell_centers[:, 0]))
            minval = plotmesh_twosides.cell_centers[minarg, 0]
            M_final[np.all([meshCyl.cell_centers[:, 0] == r, meshCyl.cell_centers[:, 1] == theta, ], axis=0)] = \
            m_bothsides[plotmesh_twosides.cell_centers[:, 0] == minval]


    n = 10
    time_steps = [
        (np.abs(t0_lm) / 20, 20), # peaktime
        (np.abs(tmax_lm_offtime) / 10, 10),  # offtime
        (3e-7, n),
        (1e-6, n),
        (5e-6, n),
        (1e-5, n),
        (1e-4, n),

    ]
    prob = tdem.Simulation3DMagneticFluxDensity(
        meshCyl, survey=survey, sigmaMap=maps.IdentityMap(meshCyl), time_steps=time_steps, t0=t0_lm
    )

    fields = prob.fields(M_final)
    dclean = prob.dpred(M_final, f=fields)
    np.savetxt('synthetic_COARSE_dclean_2D_lm_sounding'+str(sounding)+'.txt', dclean)
