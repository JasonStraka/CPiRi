import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

data_dir_path = 'datasets/BLAST/train'
data_dir_path = '/T20050027/JiyuanXv/BLAST/datasets/BLAST/train'  # Update this path as needed
data_dir_path_save = 'datasets/BLAST/train'

def main(clean_cache=False):

    num_samples = 0
    for i in range(99):
        shape = tuple(np.load(data_dir_path + f'/shape_{i}_99.npy'))
        N, L = shape
        num_samples += N

    merged_data = np.memmap(data_dir_path_save + '/data.dat', mode='w+', dtype=np.float32, shape=(num_samples, L))

    print('Merging data...')
    current_index = 0
    for i in tqdm(range(99)):
        shape = tuple(np.load(data_dir_path + f'/shape_{i}_99.npy'))
        data = np.memmap(data_dir_path + f'/data_{i}_99.dat', mode='r', dtype=np.float32, shape=shape)
        merged_data[current_index:current_index + shape[0]] = data
        current_index += shape[0]

    shape = merged_data.shape
    np.save(data_dir_path_save + '/shape.npy', shape)

    print('Data merged successfully.')
    if clean_cache:
        print('Cleaning cache...')
        for i in tqdm(range(99)):
            os.remove(data_dir_path + f'/data_{i}_99.dat')
            os.remove(data_dir_path + f'/shape_{i}_99.npy')
        print('Cache cleaned.')

def parse_args():
    parser = ArgumentParser(description='Merge data files into a single memmap file.')
    parser.add_argument('--clean_cache', default=False, help='Clean cache after merging.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(clean_cache=args.clean_cache)
