import math
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import torch
from tqdm import tqdm

from utils import analyse_interaction_from_text, analyse_user_interacted_set


class BaseImplicitDataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def user_mask_items(self, user_id: int) -> set:
        raise NotImplementedError

    def user_highlight_items(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_list(self) -> list:
        raise NotImplementedError

    def get_user_ground_truth(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        raise NotImplementedError

    @property
    def train_data_len(self) -> int:
        raise NotImplementedError

    @property
    def test_data_len(self) -> int:
        raise NotImplementedError

    @property
    def user_num(self) -> int:
        raise NotImplementedError

    @property
    def item_num(self) -> int:
        raise NotImplementedError

    @property
    def test_data_df(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def test_data_np(self) -> np.array:
        raise NotImplementedError


class BaseExplicitDataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @property
    def all_test_pairs_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_test_scores_np(self) -> np.array:
        raise NotImplementedError

    @property
    def test_data_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_test_pairs_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_test_scores_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def test_data_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_train_pairs_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_train_scores_np(self) -> np.array:
        raise NotImplementedError

    @property
    def train_data_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_train_pairs_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_train_scores_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def train_data_tensor(self) -> torch.Tensor:
        raise NotImplementedError


    @property
    def train_data_len(self) -> int:
        raise NotImplementedError

    @property
    def test_data_len(self) -> int:
        raise NotImplementedError

    @property
    def user_num(self) -> int:
        raise NotImplementedError

    @property
    def item_num(self) -> int:
        raise NotImplementedError


class MLExplicitDataLoader(BaseExplicitDataLoader):
    def __init__(self, dataset_path: str, file_name: tuple, device: torch.device, has_item_pool_file: bool = False):
        super(MLExplicitDataLoader, self).__init__(dataset_path)

        self.train_data_path: str = os.path.join(self.dataset_path, file_name[0])
        self.test_data_path: str = os.path.join(self.dataset_path, file_name[1])

        self.train_df: pd.DataFrame = pd.read_csv(self.train_data_path)  # [0: 100000]
        self.test_df: pd.DataFrame = pd.read_csv(self.test_data_path)

        self._train_data: np.array = self.train_df.iloc[:, 0: 3].values.astype(np.int64)
        self._test_data: np.array = self.test_df.iloc[:, 0: 3].values.astype(np.int64)

        self.user_positive_interaction = []
        self.user_list: list = []
        self.item_list: list = []

        self._user_num = 0
        self._item_num = 0

        self.test_user_list: list = []
        self.test_item_list: list = []
        self.ground_truth: list = []

        self.has_item_pool: bool = has_item_pool_file

        # load train dataset
        with open(self.train_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()

            print('Begin analyze raw train file')
            pairs, self.user_list, self.item_list = analyse_interaction_from_text(lines, has_value=True)

            positive_pairs: list = list(filter(lambda pair: pair[2] >= 0, pairs))

            user_positive_interaction: list = analyse_user_interacted_set(positive_pairs)
            self.user_positive_interaction = user_positive_interaction

            self._train_pairs: list = pairs

            inp.close()

        # load test dataset
        with open(self.test_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()
            print('Begin analyze raw test file')

            pairs, self.test_user_list, self.test_item_list = analyse_interaction_from_text(lines)

            self.ground_truth: list = analyse_user_interacted_set(pairs)
            inp.close()

        if self.has_item_pool:
            self.item_pool_path: str = self.dataset_path + '/test_item_pool.csv'
            with open(self.item_pool_path, 'r') as inp:
                inp.readline()
                lines: list = inp.readlines()
                print('Begin analyze item pool file')
                pairs, _, _ = analyse_interaction_from_text(lines)

                self.item_pool: list = analyse_user_interacted_set(pairs)
                inp.close()

        self._user_num = max(self.user_list + self.test_user_list) + 1
        self._item_num = max(self.item_list + self.test_item_list) + 1

        self.users_tensor: torch.LongTensor = torch.LongTensor(self.user_list)
        self.users_tensor = self.users_tensor.to(device)
        self.sorted_positive_interaction = [self.user_mask_items(user_id) for user_id in self.user_list]
        self.test_users_tensor: torch.LongTensor = torch.LongTensor(self.test_user_list)
        self.test_users_tensor = self.test_users_tensor.to(device)
        self.sorted_ground_truth: list = [self.get_user_ground_truth(user_id) for user_id in self.test_user_list]

    def user_mask_items(self, user_id: int) -> set:
        """Gets the items that the user with user ID user_id has scored in the training set

        Args:
            user_id (int): user ID

        Returns:
            set: item sets
        """
        return self.user_positive_interaction[user_id]

    @property
    def all_train_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        :return:
        """
        return self.users_tensor

    @property
    def all_train_users_by_sorted_list(self) -> list:
        return self.user_list

    def user_highlight_items(self, user_id: int) -> set:
        if not self.has_item_pool:
            raise NotImplementedError('Not has item pool!')
        return self.item_pool[user_id]

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        :return:
        """
        return self.test_users_tensor

    @property
    def all_test_users_by_sorted_list(self) -> list:
        return self.test_user_list

    def get_user_ground_truth(self, user_id: int) -> set:
        """Gets a list of real item selections in the test set for the user whose user ID is user_id

        Args:
            user_id (int): 用户ID

        Returns:
            set: _description_
        """
        return self.ground_truth[user_id]

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        """Gets a list of real item selections in the test set for all users of the current dataset

        Returns:
            list: _description_
        """
        return self.sorted_ground_truth

    @property
    def get_sorted_all_train_users_positive_interaction(self) -> list:
        """Gets a list of actual item selections from the training set for all users of the current dataset

        Returns:
            list: _description_
        """
        return self.sorted_positive_interaction

    @property
    def train_data_len(self) -> int:
        return self.train_df.shape[0]

    @property
    def test_data_len(self) -> int:
        return self.test_df.shape[0]

    @property
    def user_num(self) -> int:
        return self._user_num

    @property
    def item_num(self) -> int:
        return self._item_num

    @property
    def test_data_df(self) -> pd.DataFrame:
        return self.test_df

    @property
    def train_data_df(self) -> pd.DataFrame:
        return self.train_df

    @property
    def test_data_np(self) -> np.array:
        return self._test_data

    @property
    def train_data_np(self) -> np.array:
        return self._train_data


