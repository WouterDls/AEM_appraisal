import numpy as np
import os
from src import meshes

os.getcwd()
from SimPEG import maps

depth_discretization = np.logspace(-1, 1.4, 64)
depth_discretization = depth_discretization / np.sum(depth_discretization) * 160


def compute_denominator(save=False, case='synthetic'):
    # specifics for denominator
    if case == 'synthetic':
        meshEC_generator = meshes.generate_synthetic_model_mesh
    elif case == 'field':
        meshEC_generator = meshes.generate_field_model_mesh
    else:
        raise Exception('Define your new case')
    EPSILON = np.ones((68, 41))
    sensitivity_path = 'data and simulations/sensitivities_' + case + '/' + case

    # Do computation
    sensitivity = core_computation(EPSILON=EPSILON, meshEC_generator=meshEC_generator,
                                   sensitivity_path=sensitivity_path)

    if save:
        np.save('data and simulations/' + case + '_total_sensitivity.txt', sensitivity)
    return sensitivity


def compute_numerator(dobs_square, dpred_2D_square, rel_err, save=False, case='synthetic', modelling='perfect'):
    # specifics for denominator
    meshEC_generator = meshes.generate_synthetic_model_mesh
    EPSILON = 1 / (rel_err * dobs_square) * (dobs_square - dpred_2D_square)
    sensitivity_path = 'data and simulations/sensitivities_' + case + '/' + case

    # Do computation
    sensitivity = core_computation(EPSILON=EPSILON, meshEC_generator=meshEC_generator,
                                   sensitivity_path=sensitivity_path)
    if save:
        np.save('data and simulations/' + case + '_weighted_sensitivity' + modelling + '.txt', sensitivity)
    return sensitivity


def core_computation(EPSILON, meshEC_generator, sensitivity_path):
    meshEC = meshEC_generator()
    Jmesh_HM, Jmesh_HM_2D = meshes.generate_HM_Jacobian_computation_mesh()
    Jmesh_LM, Jmesh_LM_2D = meshes.generate_LM_Jacobian_computation_mesh()
    del Jmesh_HM, Jmesh_LM

    sensitivity = np.zeros(meshEC.n_cells)
    for t in range(41):
        for sounding in range(1, 68):
            meshEC = meshEC_generator()
            meshEC.x0 = [-np.sum(np.r_[50, np.ones(66) * 20, 50][:(sounding)]) - np.r_[50, np.ones(66) * 20, 50][
                (sounding - 1)] / 2, -np.sum(depth_discretization)]
            if t >= 18:
                J = np.loadtxt(
                    sensitivity_path + '_J_HM_sounding_' + str(sounding))
                mapping_sensitivity_to_model = maps.Mesh2Mesh([meshEC, Jmesh_HM_2D])
                sensitivity = sensitivity + np.abs(mapping_sensitivity_to_model * J[:, t - 18]) * EPSILON[sounding, t]
            if t < 18:
                J = np.loadtxt(
                    sensitivity_path + '_J_LM_sounding_' + str(sounding))
                mapping_sensitivity_to_model = maps.Mesh2Mesh([meshEC, Jmesh_LM_2D])
                sensitivity = sensitivity + np.abs(mapping_sensitivity_to_model * J[:, t]) * EPSILON[sounding, t]
    return sensitivity
