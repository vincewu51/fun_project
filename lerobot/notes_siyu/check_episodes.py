import h5py
import os
from pathlib import Path

datasets_dir = './datasets'

# Get all .hdf5 files in the datasets directory (not subdirectories)
hdf5_files = sorted([f for f in os.listdir(datasets_dir) if f.endswith('.hdf5') and os.path.isfile(os.path.join(datasets_dir, f))])

if not hdf5_files:
    print(f"No .hdf5 files found in {datasets_dir}")
else:
    print(f"Found {len(hdf5_files)} HDF5 file(s) in {datasets_dir}:\n")

    for filename in hdf5_files:
        filepath = os.path.join(datasets_dir, filename)
        try:
            with h5py.File(filepath, 'r') as f:
                episodes = []
                if 'data' in f:
                    episodes = list(f['data'].keys())
                else:
                    episodes = [k for k in f.keys() if k.startswith('demo_') or k.startswith('episode_')]

                num_total = len(episodes)
                num_success = 0

                # Check each episode for success flag
                for ep_name in episodes:
                    try:
                        if 'data' in f:
                            ep_group = f['data'][ep_name]
                        else:
                            ep_group = f[ep_name]

                        # Try different possible success flag locations
                        success = False
                        if 'success' in ep_group.attrs:
                            success = ep_group.attrs['success']
                        elif 'success' in ep_group:
                            success = bool(ep_group['success'][()])
                        elif 'is_success' in ep_group.attrs:
                            success = ep_group.attrs['is_success']
                        elif 'is_success' in ep_group:
                            success = bool(ep_group['is_success'][()])

                        if success:
                            num_success += 1
                    except Exception as e:
                        # If we can't read success flag, skip this episode
                        pass

                print(f"{filename}")
                print(f"  Total episodes: {num_total}")
                print(f"  Successful episodes: {num_success}")
                if num_total > 0:
                    print(f"  Success rate: {num_success/num_total*100:.1f}%")
                if num_total >= 2:
                    print(f"  Last 2 episodes indices: {num_total-2} {num_total-1}")
                print()
        except Exception as e:
            print(f"{filename}")
            print(f"  Error reading file: {e}")
            print()
