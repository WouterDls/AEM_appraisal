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

TIMES_HM_SKYTEM = np.r_[
    5.021000000000000208e-05, 5.571000000000000024e-05, 6.271999999999999581e-05, 7.171000000000000291e-05,
    8.320999999999999783e-05, 9.770999999999999792e-05, 1.156999999999999934e-04, 1.381999999999999982e-04,
    1.722000000000000115e-04, 2.151999999999999996e-04, 2.581999999999999877e-04, 3.012000000000000029e-04,
    3.656999999999999986e-04, 4.516999999999999748e-04, 5.592000000000000399e-04, 7.091999999999999997e-04,
    9.016999999999999625e-04, 1.137999999999999906e-03, 1.436999999999999909e-03, 1.820999999999999919e-03,
    2.292000000000000200e-03, 2.847999999999999837e-03, 3.533999999999999815e-03]

altitude = 40
radius = 10.415373695237049


cs, ncx, ncz, npad = 10, 10.0, 10.0, 10
hx = [(cs, ncx), (cs, npad, 1.25)]
ncp = 22
dp = 2 * np.pi / ncp  # cell width phi
hp =  [(dp, ncp)]
temp_belowsurface = np.logspace(np.log10(4), np.log10(24.0), 10)
temp_above_surface = temp_belowsurface[np.cumsum(temp_belowsurface) < altitude/2]
npad = 5
temp_pad = temp_belowsurface[-1] * 1.2 ** np.arange(npad)
temp_padabove = temp_above_surface[-1] * 1.4 **np.arange(npad)
hz = np.r_[temp_pad[::-1], temp_belowsurface[::-1], temp_above_surface,temp_above_surface[::-1],temp_above_surface, temp_padabove]

meshCyl = discretize.CylMesh([hx, hp, hz],)
meshCyl.x0 = [0,0, -np.sum(np.r_[temp_pad[::-1], temp_belowsurface[::-1],])]


print("the mesh has {} cells".format(meshCyl.nC))


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


for sounding in np.arange(start, start + 68):
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
        locations=np.r_[13.2, 0, rx_z], times=TIMES_HM_SKYTEM, orientation='z',
    )

    src = tdem.Src.CircularLoop(
        receiver_list=[rx],
        loc=np.r_[0.0, 0.0, src_z],  # average height as stated in DATA REPORT,
        orientation='z',
        radius=radius,
        N=1,  # number of turns
        srcType='inductive',
        waveform=waveform_hm
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


    time_steps = [
        (np.abs(t0_hm) / 20, 20), # peaktime
        (np.abs(tmax_hm_offtime) / 20, 20),  # offtime
        (3e-7, 10),
        (1e-6, 10),
        (1e-5, 10),
        (1e-4, 10),
        (5e-4, 10),

    ]
    prob = tdem.Simulation3DMagneticFluxDensity(
        meshCyl, survey=survey, sigmaMap=maps.IdentityMap(meshCyl), time_steps=time_steps, t0=t0_hm
    )

    fields = prob.fields(M_final)
    dclean = prob.dpred(M_final, f=fields)
    np.savetxt('synthetic_COARSE_dclean_2D_hm_sounding'+str(sounding)+'.txt', dclean)

