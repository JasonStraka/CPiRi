import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))
import inspect
import json
import logging
from typing import List
import numpy as np
from basicts.data import BaseDataset


class HiddenDataset(BaseDataset):

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int, output_len: int, \
        overlap: bool = False, logger: logging.Logger = None) -> None:
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap)
        self.logger = logger

        # self.data_file_path = f'datasets/{dataset_name}/data.dat' # [B N 3]
        self.data_hidden_file_path = f'datasets/{dataset_name}/data_hidden.dat' # (33601, 768, 207)
        self.data_target_file_path = f'datasets/{dataset_name}/data_target.dat' # (33601, 336, 207)
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.data_x, self.data_y = self._load_data()
        logger.info(f'Data shape: {self.data_x.shape} {self.data_y.shape}')

    def _load_description(self) -> dict:
        """
        Loads the description of the dataset from a JSON file.

        Returns:
            dict: A dictionary containing metadata about the dataset, such as its shape and other properties.

        Raises:
            FileNotFoundError: If the description file is not found.
            json.JSONDecodeError: If there is an error decoding the JSON data.
        """

        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:
        """
        Loads the time series data from a file and splits it according to the selected mode.

        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """

        try:
            _L = 33601
            # _L = 664
            data_hidden = np.memmap(self.data_hidden_file_path, dtype='float32', mode='r', shape=(_L, 768, 207))
            data_target = np.memmap(self.data_target_file_path, dtype='float32', mode='r', shape=(_L, 336, 207))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_hidden_file_path} {e}') from e

        total_len = len(data_hidden)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len#  - self.input_len + 1
        # print(f"Data lengths - Train: {train_len}, Valid: {valid_len}, Test: {test_len}")

        if self.mode == 'train':
            return data_hidden[:train_len - self.input_len + 1].copy(), data_target[:train_len - self.input_len + 1].copy()
            return data_hidden[:train_len].copy(), data_target[:train_len].copy()
        elif self.mode == 'valid':
            return data_hidden[train_len: train_len + valid_len].copy(), data_target[train_len: train_len + valid_len].copy()
        else:  # self.mode == 'test'
            return data_hidden[train_len + valid_len:].copy(), data_target[train_len + valid_len:].copy()

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        history_data = self.data_x[index]
        future_data = self.data_y[index]
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self.data_x)
