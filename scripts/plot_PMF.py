from os import path
import numpy as np
import matplotlib.pyplot as plt
import h5py
from extract_traj import DataAgent

def get_PMF(peq):
    temp = -np.log(peq)
    return temp - temp.min()

# Initialize the data agent
root_folder = "./TrajLearn/data"
host = "harmonic_well" # double_well, flat_well, harmonic_well
iter_idx_n = 0
n_frames = 10000 # 100000 (high resolution), 10000 (low resolution)

traj_lst = ['traj1', 'traj2', 'traj3']
k_max = 10 # max iteraion-id
d_data = {'xref': None, 'Vref': None, 'V_traj': None, 'V_0': None}

# Load potential used in simulation
f_jld = path.join(root_folder, host, "potential_peq.jld")
jld = h5py.File(f_jld, "r")
d_data['xref'] = np.array(jld["xref"])[0]
d_data['Vref'] = np.array(jld["Vref"])[0]
d_data['Vref'] = d_data['Vref'] - d_data['Vref'].min()
jld.close()

f_jld = path.join(root_folder, host, f"peq_update_record_n{iter_idx_n}_{n_frames}.jld")
jld = h5py.File(f_jld, "r")
p_container = np.array(jld["p_container"])
d_data['V_traj'] = get_PMF(p_container[:, k_max])
d_data['V_0'] = get_PMF(p_container[:, 0])
jld.close()

# Plotting Variable
figsize = (2.33, 1.5)
lbfz = 8
lgfz = 8
tickfz = 6
lw = 1.0 # linewidth

# lim, ticks
xlim = (-64, 64)
xticks = np.arange(-64, 65, 16)
ylim = (-0.5, 10)
yticks = range(0, 11, 2)

dpi = 100
out = path.join(".", "figures", f"{host}_PMF.{n_frames}.png")

fig, ax = plt.subplots(figsize=figsize)
ax.plot(d_data['xref'], d_data['Vref'], '--', linewidth=0.75, label=r"$V(x)$", color="red", alpha=0.9)
ax.plot(d_data['xref'], d_data['V_0'], linewidth=0.75, label=r"$V^{\mathrm{(0)}}(x)$", color="#ffae49", alpha=0.8)
ax.plot(d_data['xref'], d_data['V_traj'], linewidth=0.75, label= r'$V^{(' + f'{k_max}' + ')}(x)$',
        color="darkcyan", alpha=0.4)

ax.set_xlabel('$x$ (Å)', fontsize=lbfz)
ax.set_ylabel(r'$V^{(m)}(x)$ ($k_{\mathrm{B}}\mathrm{T}$)', fontsize=lbfz)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.tick_params(axis="both", labelsize=tickfz)
ax.legend(frameon=False, fontsize=lgfz)
plt.subplots_adjust(top = 0.9, bottom = 0.22, right = 0.96, left = 0.2, hspace = 0, wspace = 0)
plt.savefig(out, transparent=False)
plt.show()