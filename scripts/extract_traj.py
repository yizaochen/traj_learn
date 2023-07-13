import sys
import traceback
from os import path
import h5py
import numpy as np

class DataAgent:

    def __init__(self, root_folder, host):
        self.root_folder = root_folder
        self.host = host
        self.data_folder = path.join(root_folder, host)
        self.f_h5 = path.join(self.data_folder, 'traj.jld')
        self.raw_data = self._get_raw_data()

    def make_h5(self, n_frames):
        interval = self._get_interval(n_frames)
        out = path.join(self.data_folder, f'traj.{n_frames}.frames.hdf5')
        f = h5py.File(out, 'w')
        time_array = self.raw_data['time'][::interval] # unit is T, T = 0.1 ns
        f['time'] = time_array
        f['distance'] = self.raw_data['distance'][::interval]
        delta_t = time_array[1] - time_array[0]
        print(f'[INFO] Interval: {interval} Delta t:{delta_t:.2E} T  N_frames: {n_frames}')
        f.close()

    def read_h5(self, n_frames):
        in_file = path.join(self.data_folder, f'traj.{n_frames}.frames.hdf5')
        f = h5py.File(in_file, 'r')
        data = dict()
        data['time'] = np.array(f['time'])
        data['distance'] = np.array(f['distance'])
        f.close()
        return data

    def _get_raw_data(self):
        data = dict()
        f = h5py.File(self.f_h5, 'r')
        data['time'] = np.array(f['t_record'])[0,:-1]
        data['distance'] = np.array(f['y_record'])[0,:-1]
        f.close()
        return data
    
    def _get_interval(self, n_frames):
        total_in_raw = self.raw_data['time'].shape[0]
        interval = int(total_in_raw / n_frames)
        if self.raw_data['time'][::interval].shape[0] != n_frames:
            traceback.print_exc()
            sys.exit("[Error] The trajectory corrupted.")
        return interval
    

if __name__ == '__main__':
    root_folder = "./TrajLearn/data"
    host = "harmonic_well" # double_well, flat_well, harmonic_well
    d_agent = DataAgent(root_folder, host)

    n_frames = 10000 # number of frames wanted in the new trajectory
    d_agent.make_h5(n_frames) 
