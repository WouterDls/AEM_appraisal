import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pandas as pd

os.getcwd()
import discretize
from SimPEG import maps
from matplotlib.cm import get_cmap

depth_discretization = np.logspace(-1, 1.4, 64)
depth_discretization = depth_discretization / np.sum(depth_discretization) * 160

def generate_synthetic_model_mesh():
    mesh = discretize.TensorMesh([np.r_[50, np.ones(66) * 20, 50], np.flip(depth_discretization)], x0=[0, -160])
    return mesh

def generate_field_model_mesh():
    try:
        cell_width = np.loadtxt('../altitudes_per_sounding.txt')[1:-1]
    except:
        raise Exception('The field data is available at DOI [INSERT].')
    mesh = discretize.TensorMesh([np.r_[500, cell_width, 500], np.flip(depth_discretization)], )
    return mesh


def generate_HM_Jacobian_computation_mesh():
    """
    Generates meshes to compute the Jacobian on the coarse mesh, optimized for the HM waveform
    :return:
    """

    csx = 5  # cell size for the horizontal direction
    csz = 2.5  # cell size for the vertical direction
    pf = 1.05  # expansion factor for the padding cells
    pf2 = 1.2

    npadx = 25  # number of padding cells in the x-direction
    npady = 8  # number of padding cells in the y-direction
    npadz = 20  # number of padding cells in the z-direction

    core_domain_x = np.r_[-5, 5]  # extent of uniform cells in the x-direction
    core_domain_z = np.r_[-40.0, 0.0]  # extent of uniform cells in the z-direction

    # number of cells in the core region
    ncx = int(np.diff(core_domain_x) / csx)
    ncz = int(np.diff(core_domain_z) / csz)

    # create a 3D tensor mesh
    mesh_HM_3D = discretize.TensorMesh(
        [
            [(csx, npadx, -pf), (csx, ncx), (csx, npadx, pf)],
            [(csx, npady, -pf2), (csx, 1), (csx, npady, pf2)],
            [(csz, npadz, -pf), (csz, ncz), (csz, npadz // 2, pf2)],
        ]
    )
    # set the origin
    mesh_HM_3D.x0 = np.r_[
        -mesh_HM_3D.hx.sum() / 2.0, -mesh_HM_3D.hy.sum() / 2.0, -mesh_HM_3D.hz[: npadz + ncz].sum()
    ]

    print("the mesh has {} cells".format(mesh_HM_3D.nC))

    mesh_LM_2D = discretize.TensorMesh([mesh_HM_3D.hx, mesh_HM_3D.hz[mesh_HM_3D.vectorCCz <= 0]])
    mesh_LM_2D.x0 = [-mesh_LM_2D.hx.sum() / 2.0, -mesh_LM_2D.hy.sum()]

    return mesh_HM_3D, mesh_LM_2D


def generate_LM_Jacobian_computation_mesh():
    """
    Generates meshes to compute the Jacobian on the coarse mesh, optimized for the LM waveform
    :return:
    """
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
    mesh_LM_3D = discretize.TensorMesh(
        [
            [(csx, npadx, -pf), (csx, ncx), (csx, npadx, pf)],
            [(csx, npady, -pf2), (csx, 1), (csx, npady, pf2)],
            [(csz, npadz, -pf), (csz, ncz), (csz, npadz, pf2)],
        ]
    )
    # set the origin
    mesh_LM_3D.x0 = np.r_[
        -mesh_LM_3D.hx.sum() / 2.0, -mesh_LM_3D.hy.sum() / 2.0, -mesh_LM_3D.hz[: npadz + ncz].sum()
    ]

    mesh_LM_2D = discretize.TensorMesh([mesh_LM_3D.hx, mesh_LM_3D.hz[mesh_LM_3D.vectorCCz <= 0]])
    mesh_LM_2D.x0 = [-mesh_LM_2D.hx.sum() / 2.0, -mesh_LM_2D.hy.sum()]

    return mesh_LM_3D, mesh_LM_2D
