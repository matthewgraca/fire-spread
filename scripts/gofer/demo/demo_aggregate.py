from gofer.temporal_downsampler import aggregate
import pickle

with open('temp/metadata.pkl', 'rb') as f:
    data = pickle.load(f)
    dates = data['dates']

east_ds = aggregate(
    goes_save_dir='/home/mgraca/Workspace/fire-spread/data/goes',
    csv_path='/home/mgraca/Workspace/fire-spread/temp/east_files.csv',
    dates=dates
)

print(east_ds)
