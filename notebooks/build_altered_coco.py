#this script should not work on its own, it was copied and pasted from a notebook

from radbm.utils.fetch import fetch_file
def coco_finder(which, path=None):
    file_paths = fetch_file(
        which,
        path,
        data_type='dataset',
        subdirs=['MSCoco', 'Coco'],
        download=False
    )
    if not file_paths:
        raise FileNotFoundError('could not locate squared_coco.hdf5')
    return h5py.File(file_paths[0], 'r')

import h5py
root = os.environ['SLURM_TMPDIR'] + '/datasets/Coco/{}'
with coco_finder('squared_coco.hdf5') as coco:
    N = len(coco['data'])
    train_indexes = list(range(60000,N-40000))
    valid_indexes = list(range(0,60000)) + list(range(N-40000,N))
    query_train_indexes = train_indexes[:10000]
    query_valid_indexes = valid_indexes[:10000]
    with h5py.File(root.format('altered_coco.hdf5'), 'w') as altered_coco:
        train_data = altered_coco.create_dataset('train', (10000,3,256,256), dtype=np.uint8)
        valid_data = altered_coco.create_dataset('valid', (10000,3,256,256), dtype=np.uint8)
        train_coco_id = altered_coco.create_dataset('train_coco_id', (10000,), dtype=np.uint64, data=coco['coco_id'][60000:70000])
        valid_coco_id = altered_coco.create_dataset('valid_coco_id', (10000,), dtype=np.uint64, data=coco['coco_id'][:10000])
        train_gen = torch.Generator('cuda').manual_seed(0xcafe1)
        valid_gen = torch.Generator('cuda').manual_seed(0xcafe2)
        train_alter = AlterImages((100, 3, 256, 256), generator=train_gen)
        valid_alter = AlterImages((100, 3, 256, 256), generator=valid_gen)
        for i in range(10000//100):
            print(i, end='\r')
            train_im = torch.tensor(coco['data'][60000+100*i:60000+100*(i+1)], device='cuda').float()/255
            valid_im = torch.tensor(coco['data'][100*i:100*(i+1)], device='cuda').float()/255
            train_altered_im = (train_alter(train_im)*255).to(torch.uint8).cpu().numpy()
            valid_altered_im = (valid_alter(valid_im)*255).to(torch.uint8).cpu().numpy()
            train_data[100*i:100*(i+1)] = train_altered_im
            valid_data[100*i:100*(i+1)] = valid_altered_im