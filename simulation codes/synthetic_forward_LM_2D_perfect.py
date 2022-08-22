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

TIMES_LM_SKYTEM = np.logspace(-6,-3,100)

altitude = 40
radius = 10.415373695237049

cs, ncx, ncz, npad = 0.5, 50.0, 10.0, 43
hx = [(cs, ncx), (cs, npad, 1.15)]
ncp = 45
dp = 2 * np.pi / ncp  # cell width phi
hp =  [(dp, ncp)]

npad = 18
temp_belowsurface = np.logspace(np.log10(.05), np.log10(12.0), 100)
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



##
def wave_function_lm(t):
    x = np.r_[t0_lm, LM_waveform['Time'].values, 1e-1]
    y = np.r_[0, LM_waveform['Current'].values, 0]
    f = interp1d(x, y)
    return f(t)

waveform_lm = tdem.sources.RawWaveform(waveFct=wave_function_lm, offTime=0.)

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
        locations=np.r_[13.2, 0, rx_z], times=TIMES_LM_SKYTEM, orientation='z',
    )

    src = tdem.Src.CircularLoop(
        receiver_list=[rx],
        loc=np.r_[0.0, 0.0, src_z],  # average height as stated in DATA REPORT,
        orientation='z',
        radius=radius,
        N=1,  # number of turns
        # srcType='inductive',
        # waveform=waveform_lm
    )

    srcList.append(src)

    survey = tdem.Survey(srcList)


    plotmesh_twosides = discretize.TensorMesh(
        [
            np.r_[np.flip(meshCyl.hx), meshCyl.hx], meshCyl.hz
        ], x0=[-np.sum(meshCyl.hx), meshCyl.x0[2]]
    )
    mapping = maps.Mesh2Mesh([plotmesh_twosides, tussenmesh])
    m_bothsides=mapping*(mtest.flatten(order='F'))

    M_final = np.zeros(meshCyl.n_cells)
    for r in meshCyl.vectorCCx:
        for theta in meshCyl.vectorCCy:
            minarg = np.argmin(np.abs(r * np.cos(theta) - plotmesh_twosides.cell_centers[:, 0]))
            minval = plotmesh_twosides.cell_centers[minarg, 0]
            M_final[np.all([meshCyl.cell_centers[:, 0] == r, meshCyl.cell_centers[:, 1] == theta, ], axis=0)] = \
            m_bothsides[plotmesh_twosides.cell_centers[:, 0] == minval]

    n = 50
    time_steps = [
        #(np.abs(t0_lm) / 20, 20), # peaktime
        #(np.abs(tmax_lm_offtime) / 10, 10),  # offtime
        (3e-7, n),(3e-7, n),
        (1e-6, n),(1e-6, n),
        (1e-6, n),(1e-6, n),
        (1e-6, n), (1e-6, n),
        (1e-6, n), (1e-6, n),
        (2e-6, n),(2e-6, n),
        (2e-5, n), (2e-5, n),

    ]
    prob = tdem.Simulation3DMagneticFluxDensity(
        meshCyl, survey=survey, sigmaMap=maps.IdentityMap(meshCyl), time_steps=time_steps, #t0=t0_lm
    )

    fields = prob.fields(M_final)
    dclean = prob.dpred(M_final, f=fields)
    np.savetxt('synthetic_PERFECT_dclean_2D_lm_sounding'+str(sounding)+'.txt', dclean)



##
"""
plotmesh_twosides.plot_image(m_bothsides)
plt.show()

##

##
plotmesh = discretize.TensorMesh(
    [
        np.r_[ meshCyl.hx], meshCyl.hz
    ], x0=[0, meshCyl.x0[2]]
)

for i in np.arange(20):
    selection = meshCyl.cell_centers[:,1] == meshCyl.vectorCCy[i]
    plt.figure()
    plotmesh.plot_image(M_final[selection],  pcolor_opts={"cmap": "coolwarm", })
    plt.title(str(meshCyl.vectorCCy[i]/(np.pi*2)*360))
    #plt.ylim(4500,5000)
    #plt.xlim(0, 1000)
    plt.show()
##
"""