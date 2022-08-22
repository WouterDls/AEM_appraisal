#! /usr/bin/env python
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pymatsolver import Pardiso as Solver
from SimPEG.utils import mkvc
import discretize
from SimPEG import (
    maps,
)
from SimPEG.electromagnetics import time_domain as tdem


# Setup
# -----
sigma_air = 1e-8
radius = 10.415373695237049
coil_separation = radius
LINE = 306025
depth_discretization = np.logspace(-1, 1.4, 64)
depth_discretization = depth_discretization / np.sum(depth_discretization) * 160

times_lm = np.r_[
    1.033999999999999926e-05, 1.285000000000000064e-05, 1.633999999999999941e-05, 2.084000000000000037e-05,
    2.633999999999999853e-05, 3.334999999999999748e-05, 4.233999999999999781e-05, 5.383999999999999951e-05,
    6.833999999999999960e-05, 8.634000000000000344e-05, 1.088000000000000019e-04, 1.428000000000000016e-04,
    1.857999999999999897e-04, 2.288000000000000049e-04, 2.717999999999999930e-04, 3.362999999999999887e-04,
    4.223000000000000191e-04, 5.297999999999999758e-04,]

TIMES_SKYTEM = np.r_[1.034e-05, 1.285e-05, 1.634e-05, 2.084e-05, 2.634e-05, 3.335e-05,
                     4.234e-05, 5.021e-05, 5.384e-05, 5.571e-05, 6.272e-05, 6.834e-05,
                     7.171e-05, 8.321e-05, 8.634e-05, 9.771e-05, 1.088e-04, 1.157e-04,
                     1.382e-04, 1.428e-04, 1.722e-04, 1.858e-04, 2.152e-04, 2.288e-04,
                     2.582e-04, 2.718e-04, 3.012e-04, 3.363e-04, 3.657e-04, 4.223e-04,
                     4.517e-04, 5.298e-04, 5.592e-04, 7.092e-04, 9.017e-04, 1.138e-03,
                     1.437e-03, 1.821e-03, 2.292e-03, 2.848e-03, 3.534e-03]

for sounding in np.arange(1, 68):
    csx = 2  # cell size for the horizontal direction
    csz = 2  # cell size for the vertical direction
    pf = 1.1  # expansion factor for the padding cells
    pf2 = 1.2

    npadx = 20  # number of padding cells in the x-direction
    npady = 10  # number of padding cells in the y-direction
    npadz = 10  # number of padding cells in the z-direction

    core_domain_x = np.r_[-5, 5]  # extent of uniform cells in the x-direction
    core_domain_z = np.r_[-40.0, 0.0]  # extent of uniform cells in the z-direction

    # number of cells in the core region
    ncx = int(np.diff(core_domain_x) / csx)
    ncz = int(np.diff(core_domain_z) / csz)

    # create a 3D tensor mesh
    mesh = discretize.TensorMesh(
        [
            [(csx, npadx, -pf), (csx, ncx), (csx, npadx, pf)],
            [(csx, npady, -pf2), (csx, 1), (csx, npady, pf2)],
            [(csz, npadz, -pf), (csz, ncz), (csz, npadz, pf2)],
        ]
    )
    # set the origin
    mesh.x0 = np.r_[
        -mesh.hx.sum() / 2.0, -mesh.hy.sum() / 2.0, -mesh.hz[: npadz + ncz].sum()
    ]

    print("the mesh has {} cells".format(mesh.nC))
    mesh.plotGrid()
    plt.show()


    ###############################################################################
    # Inversion Mesh
    # --------------
    #
    # Here, we set up a 2D tensor mesh which we will represent the inversion model
    # on

    inversion_mesh = discretize.TensorMesh([mesh.hx, mesh.hz[mesh.vectorCCz <= 0]])
    inversion_mesh.x0 = [-inversion_mesh.hx.sum() / 2.0, -inversion_mesh.hy.sum()]


    m = np.loadtxt('data and simulations/field_model_selection_quasi2D.txt')
    m = np.flip(m.reshape(68, 64, ), axis=1).reshape(-1, 1, order='F')
    try:
        d = np.loadtxt('../distances_between_soundings.txt')
        cell_width = d[1:-1]
        altitudes = np.loadtxt('../altitudes_per_sounding.txt')
    except:
        raise Exception('The field data is available at DOI [INSERT].')
    plotregMesh = discretize.TensorMesh([np.r_[500, cell_width, 500], np.flip(depth_discretization)], )
    plotregMesh.x0 = [-np.sum(np.r_[500, cell_width, 500][:sounding]) + np.r_[500, cell_width, 500][sounding] / 2,
                      -np.sum(x)]

    mapping = maps.Mesh2Mesh([inversion_mesh, plotregMesh])

    # create a 2D mesh that includes air cells
    mesh2D = discretize.TensorMesh([mesh.hx, mesh.hz], x0=mesh.x0[[0, 2]])
    active_inds = mesh2D.gridCC[:, 1] < 0  # active indices are below the surface

    mapping = (
            maps.Surject2Dto3D(mesh)
            * maps.InjectActiveCells(  # populates 3D space from a 2D model
        mesh2D, active_inds, sigma_air)
            * maps.ExpMap(  # adds air cells
        nP=inversion_mesh.nC)  # takes the exponential (log(sigma) --> sigma)
    )
    mapping_to_inversion_mesh = maps.Mesh2Mesh([inversion_mesh, plotregMesh])
    m_inversion_mesh = mapping_to_inversion_mesh * m


    lm_waveform_time = np.r_[-8.00000e-04, -7.89693e-04, -7.73840e-04, -7.17888e-04,
                             -6.03188e-04, -4.52118e-04, -1.86348e-04, -3.06160e-05,
                             0.00000e+00, 7.00000e-08, 1.50000e-07, 3.30000e-07,
                             5.10000e-07, 1.01000e-06, 1.19000e-06, 1.37000e-06,
                             1.70000e-06, 2.06000e-06, 2.27000e-06, 2.50000e-06,
                             3.03000e-06, 3.70000e-06, 5.24000e-06]

    lm_waveform_current = np.r_[0., 0.0232847, 0.0554015, 0.142117, 0.294672,
                                0.474526, 0.776423, 0.967518, 1., 0.99526,
                                0.976507, 0.901233, 0.798214, 0.482028, 0.382865,
                                0.296338, 0.17208, 0.0824201, 0.0480014, 0.0219084,
                                -0.00693382, -0.0123619, 0.]

    data_lm = {'Time': lm_waveform_time, 'Current': lm_waveform_current}
    lm_waveform = pd.DataFrame(data_lm)

    LM_waveform = pd.read_excel('LM_waveform.xls')
    t0_lm = min(LM_waveform['Time'].values)


    def wave_function_lm(t):
        x = np.r_[t0_lm, LM_waveform['Time'].values, 1e-1]
        y = np.r_[0, LM_waveform['Current'].values, 0]
        f = interp1d(x, y)
        return f(t)


    waveform_lm = tdem.sources.RawWaveform(waveFct=wave_function_lm, offTime=0.)

    src_locations = np.cumsum(np.r_[0, d]) - np.sum(d) / 2
    # create our source list - one source per location
    srcList = []
    x = 0
    src_loc = np.r_[x, 0.0, altitudes[sounding]]
    rx_loc = np.r_[x - 13.2, 0.0, altitudes[sounding] + 2]

    rx = tdem.Rx.PointMagneticFluxTimeDerivative(
        locations=rx_loc, times=times_lm, orientation='z',
    )

    # Default step-off
    src = tdem.Src.CircularLoop(
        receiver_list=[rx],
        location=src_loc,  # average height as stated in DATA REPORT,
        orientation='z',
        radius=radius,
        N=1,  # number of turns
        srcType='inductive',
        waveform=waveform_lm
    )
    srcList.append(src)

    # create the survey and problem objects for running the forward simulation
    survey = tdem.Survey(srcList)

    i = 20
    time_steps = [
        (np.abs(t0_lm) / 10, 10), (5e-7, i), (1e-6, i), (1e-5, i), (5e-5, 10),
    ]
    # prob = tdem.Simulation3DMagneticFluxDensity(
    prob = tdem.Simulation3DElectricField(
        mesh, survey=survey, sigmaMap=mapping, time_steps=time_steps, t0=t0_lm, solver=Solver
    )


    ##
    """
    Sensitivies
    """
    f = prob.fields(m_inversion_mesh)
    J = []
    for i in np.arange(0, m_inversion_mesh.size):
        v = np.zeros_like(m_inversion_mesh)
        v[i] = 1.

        """
        Jvec computes the sensitivity times a vector

        .. math::
            \mathbf{J} \mathbf{v} = \\frac{d\mathbf{P}}{d\mathbf{F}} \left(
            \\frac{d\mathbf{F}}{d\mathbf{u}} \\frac{d\mathbf{u}}{d\mathbf{m}} +
            \\frac{\partial\mathbf{F}}{\partial\mathbf{m}} \\right) \mathbf{v}

        where

        .. math::
            \mathbf{A} \\frac{d\mathbf{u}}{d\mathbf{m}} +
            \\frac{\partial \mathbf{A}(\mathbf{u}, \mathbf{m})}
            {\partial\mathbf{m}} =
            \\frac{d \mathbf{RHS}}{d \mathbf{m}}
        """

        ftype = prob._fieldType + "Solution"  # the thing we solved for
        prob.model = m_inversion_mesh

        # mat to store previous time-step's solution deriv times a vector for
        # each source
        # size: nu x nSrc

        # this is a bit silly

        # if self._fieldType == 'b' or self._fieldType == 'j':
        #     ifields = np.zeros((self.mesh.nF, len(Srcs)))
        # elif self._fieldType == 'e' or self._fieldType == 'h':
        #     ifields = np.zeros((self.mesh.nE, len(Srcs)))

        # for i, src in enumerate(self.survey.source_list):
        dun_dm_v = np.hstack(
            [
                mkvc(prob.getInitialFieldsDeriv(src, v, f=f), 2)
                for src in prob.survey.source_list
            ]
        )
        # can over-write this at each timestep
        # store the field derivs we need to project to calc full deriv
        df_dm_v = prob.Fields_Derivs(prob)

        Adiaginv = None

        for tInd, dt in zip(range(prob.nT), prob.time_steps):
            # keep factors if dt is the same as previous step b/c A will be the
            # same
            if Adiaginv is not None and (tInd > 0 and dt != prob.time_steps[tInd - 1]):
                Adiaginv.clean()
                Adiaginv = None

            if Adiaginv is None:
                A = prob.getAdiag(tInd)
                Adiaginv = prob.solver(A, **prob.solver_opts)

            Asubdiag = prob.getAsubdiag(tInd)

            for i, src in enumerate(prob.survey.source_list):

                # here, we are lagging by a timestep, so filling in as we go
                for projField in set([rx.projField for rx in src.receiver_list]):
                    df_dmFun = getattr(f, "_%sDeriv" % projField, None)
                    # df_dm_v is dense, but we only need the times at
                    # (rx.P.T * ones > 0)
                    # This should be called rx.footprint

                    df_dm_v[src, "{}Deriv".format(projField), tInd] = df_dmFun(
                        tInd, src, dun_dm_v[:, i], v
                    )

                un_src = f[src, ftype, tInd + 1]

                # cell centered on time mesh
                dA_dm_v = prob.getAdiagDeriv(tInd, un_src, v)
                # on nodes of time mesh
                dRHS_dm_v = prob.getRHSDeriv(tInd + 1, src, v)

                dAsubdiag_dm_v = prob.getAsubdiagDeriv(tInd, f[src, ftype, tInd], v)

                JRHS = dRHS_dm_v - dAsubdiag_dm_v - dA_dm_v

                # step in time and overwrite
                if tInd != len(prob.time_steps + 1):
                    dun_dm_v[:, i] = Adiaginv * (JRHS.flatten() - Asubdiag * dun_dm_v[:, i])

        Jv = []
        for src in prob.survey.source_list:
            for rx in src.receiver_list:
                Jv.append(
                    rx.evalDeriv(
                        src,
                        prob.mesh,
                        prob.time_mesh,
                        f,
                        mkvc(df_dm_v[src, "%sDeriv" % rx.projField, :]),
                    )
                )
        Adiaginv.clean()
        # del df_dm_v, dun_dm_v, Asubdiag
        # return mkvc(Jv)
        np.hstack(Jv)

        J.append(Jv)

    J = np.vstack(J)

    # np.savetxt('data and simulations/sensitivities_field/field_J_LM_sounding_'+str(sounding), J)

