from os import path
import numpy as np
import matplotlib.pyplot as plt
import h5py
from extract_traj import DataAgent

# io-parameters
root_folder = "./TrajLearn/data"
host = "harmonic_well" # double_well, flat_well, harmonic_well
n_frames = 10000 # number of frames wanted in the new trajectory
out_png = path.join(".", "figures", f"{host}_{n_frames}_traj.png")

# read Data
d_agent = DataAgent(root_folder, host)
trim_data = d_agent.read_h5(n_frames)
trim_data['time'] = trim_data['time'] * 1e-4 # convert from T to us, 1 T = 0.1 ns = 1e-4 us

# plotting parameters
figsize = (7, 1.5)
lbfz = 8
tickfz = 6
lw = 0.6

xlim = (trim_data['time'][0], trim_data['time'][-1])
xticks = range(0, 11, 2)
ylim = (-64, 64)
yticks = np.arange(-64, 65, 16)
color = "darkcyan"

# plot
fig, ax = plt.subplots(figsize=figsize)
ax.plot(trim_data['time'], trim_data['distance'], linewidth=lw, color=color, alpha=1)
ax.set_ylabel('$x$ (Å)', fontsize=lbfz)
ax.set_xlabel("time ($\mathrm{\mu}$s)", fontsize=lbfz)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.tick_params(axis="both", labelsize=tickfz)

plt.subplots_adjust(top = 0.92, bottom = 0.25, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
plt.savefig(out_png, dpi=100, transparent=False)
plt.show()