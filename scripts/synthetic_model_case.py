import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from src import meshes, appraisal

data_path = './data and simulations/'
##
"""

LOADING DATA

"""
# Loading SkyTEM waveforms, used in this example
LM_waveform = pd.read_excel(data_path +'LM_waveform.xls')
HM_waveform = pd.read_excel(data_path +'HM_waveform.xls')

m = np.loadtxt(data_path + 'synthetic_model_selection_quasi2D.txt')
m_square = np.flip(np.exp(m.reshape(68, -1)), axis=1)
recovered_model = m_square.reshape(-1, 1, order='F')

# The observed data - that is the synthetic data 2.5D data that is used to mimic a true survey.
dobs_square = pd.read_csv(data_path + 'synthetic_dobs.csv',).to_numpy()[:,1:]
rel_err_square = np.ones(dobs_square.shape)*0.03 # relative error on data
dpred_1D_square = pd.read_csv(data_path + 'synthetic_dpred_1D.csv',).to_numpy()[:,1:]
dpred_2D_square = pd.read_csv(data_path + 'synthetic_dpred_2D_perfect.csv',).to_numpy()[:,1:]

dpred_1D_imperfect_square = pd.read_csv(data_path + 'synthetic_dpred_1D_forced_imperfect.csv',).to_numpy()[:,1:]
dpred_2D_imperfect_square = pd.read_csv(data_path + 'synthetic_dpred_2D_imperfect.csv',).to_numpy()[:,1:]


##
"""
Recovered model
"""
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
im = meshes.generate_synthetic_model_mesh().plot_image(recovered_model, pcolor_opts={"cmap": 'coolwarm', }, ax=ax)
plt.ylim(-80, 0)
plt.ylabel('Depth (m)', fontsize=13)
plt.xlabel('x (m)', fontsize=13)
plt.vlines(700, -65, -25, color='k', linewidth=2, label="True interface")
plt.hlines(-25, 700, 0, color='k', linewidth=2, )
plt.hlines(-65, 700, 1400, color='k', linewidth=2, )
plt.legend()
plt.title('Recovered model', fontsize=15)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.15)
cbar = fig.colorbar(im[0], cax=cax, )
cbar.set_label('EC (S/m)', fontsize='13')  # rotation=270)
plt.tight_layout()
plt.show()

##

"""
Perfect Modelling
"""

# Read from file or do core computation
fromFile = False

if fromFile:
    N = np.loadtxt('data and simulations/synthetic_weighted_sensitivity_perfect.txt')
    D = np.loadtxt('data and simulations/synthetic_total_sensitivity.txt')
else:
    N = appraisal.compute_numerator(dobs_square, dpred_2D_square,rel_err_square, save=False, case='synthetic',modelling='perfect')
    D = appraisal.compute_denominator(save=False, case='synthetic')


##
plot_mesh = meshes.generate_synthetic_model_mesh()
plot_mesh.plot_image(N/D, pcolor_opts={"cmap": "inferno", })
plt.title('Normalised gradient - Perfect modelling')
plt.ylim(-120,0)
plt.show()
##
"""
Imperfect Modelling
"""

# Read from file or do core computation
fromFile = False

if fromFile:
    N = np.loadtxt('data and simulations/synthetic_weighted_sensitivity_imperfect.txt')
    D = np.loadtxt('data and simulations/synthetic_total_sensitivity.txt')
else:
    N = appraisal.compute_numerator(dpred_1D_imperfect_square, dpred_2D_imperfect_square,rel_err_square, save=False, case='synthetic',modelling='imperfect')
    D = appraisal.compute_denominator(save=False, case='synthetic')


##
plot_mesh = meshes.generate_synthetic_model_mesh()
plot_mesh.plot_image(N/D, pcolor_opts={"cmap": "inferno", })
plt.title('Normalised gradient - Imperfect modelling')
plt.ylim(-120,0)
plt.show()


##

