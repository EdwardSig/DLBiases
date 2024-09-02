import itertools
import math

import numpy as np
import torch
from colorama import Fore, Style
from torch import nn

from evaluate import ImplicitTestManager
from models import GeneralDebiasImplicitRecommender
from utils import mini_batch, merge_dict, _mean_merge_dict_func, transfer_loss_dict_to_line_str


class ExplicitTrainManager:
    def __init__(
            self, model: GeneralDebiasImplicitRecommender, evaluator: ImplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, pop_env_label: torch.Tensor, con_env_label: torch.Tensor,
            env_classifier: torch.Tensor,
            bias_count: int, batch_size: int, epochs: int, cluster_interval: int, evaluate_interval: int, lr: float,
            invariant_coe: float, env_aware_coe: float, env_coe: float, L2_coe: float, L1_coe: float,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            begin_cluster_epoch: int = None, stop_cluster_epoch: int = None, cluster_use_random_sort: bool = True,
            use_recommend_re_weight: bool = True
    ):
        self.model: GeneralDebiasImplicitRecommender = model
        self.evaluator: ImplicitTestManager = evaluator
        self.envs_num: int = self.model.env_num
        self.device: torch.device = device
        self.users_tensor: torch.Tensor = training_data[:, 0]
        self.items_tensor: torch.Tensor = training_data[:, 1]
        self.scores_tensor: torch.Tensor = training_data[:, 2].float()
        self.user_positive_interaction = evaluator.data_loader.user_positive_interaction
        self.pop_envs: torch.LongTensor = pop_env_label  # Used for consistency bias environment training
        self.pop_envs = self.pop_envs.to(device)
        self.con_envs: torch.LongTensor = con_env_label  # For selection bias environment training
        self.con_envs = self.con_envs.to(device)
        self.env_classifier: torch.Tensor = env_classifier  # Used for multi-label identification
        self.env_classifier = self.env_classifier.to(device)
        self.cluster_interval: int = cluster_interval
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        self.recommend_loss_type = nn.MSELoss
        self.cluster_distance_func = nn.BCELoss(reduction='none')
        self.env_loss_type = nn.BCEWithLogitsLoss

        self.invariant_coe = invariant_coe
        self.env_aware_coe = env_aware_coe
        self.env_coe = env_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(training_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        self.use_class_re_weight: bool = use_class_re_weight
        self.use_recommend_re_weight: bool = use_recommend_re_weight
        self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(training_data.shape[0])).to(device)
        self.class_weights: torch.Tensor = torch.Tensor(np.zeros(self.envs_num)).to(device)

        self.test_begin_epoch: int = test_begin_epoch

        self.begin_cluster_epoch: int = begin_cluster_epoch
        self.stop_cluster_epoch: int = stop_cluster_epoch

        self.eps_random_tensor: torch.Tensor = self._init_eps().to(self.device)

        self.cluster_use_random_sort: bool = cluster_use_random_sort

        self.const_env_tensor_list: list = []

        item_num = evaluator.data_loader.item_num

        self.const_env_tensor_list = []

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            batch_pop_envs_tensor: torch.Tensor,
            batch_con_envs_tensor: torch.Tensor,
            batch_env_classifier: torch.Tensor,
            batch_sample_weights: torch.Tensor,
            alpha,
            batch_index: int
    ) -> dict:

        # output of DLBias
        mf_score, env_aware_score, env_outputs = self.model(
            batch_users_tensor, batch_items_tensor,
            batch_pop_envs_tensor, batch_con_envs_tensor, alpha)

        env_loss = self.env_loss_type()
        recommend_loss = self.recommend_loss_type()
        mf_loss: torch.Tensor = recommend_loss(mf_score, batch_scores_tensor)
        env_aware_loss: torch.Tensor = recommend_loss(env_aware_score, batch_scores_tensor)
        envs_loss: torch.Tensor = env_loss(env_outputs, batch_env_classifier)
        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor,
                                                     batch_pop_envs_tensor, batch_con_envs_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor,
                                                     batch_pop_envs_tensor, batch_con_envs_tensor)

        """
        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe
        """

        loss: torch.Tensor = mf_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'invariant_loss': float(mf_loss),
            'env_aware_loss': float(env_aware_loss),
            'envs_loss': float(envs_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def cluster_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
    ) -> torch.Tensor:
        distances_list: list = []
        for env_idx in range(self.envs_num):
            envs_tensor: torch.Tensor = self.const_env_tensor_list[env_idx][0:batch_users_tensor.shape[0]]

            cluster_pred: torch.Tensor = self.model.cluster_predict(batch_users_tensor, batch_items_tensor, envs_tensor)

            distances: torch.Tensor = self.cluster_distance_func(cluster_pred, batch_scores_tensor)

            distances = distances.reshape(-1, 1)
            distances_list.append(distances)

        each_envs_distances: torch.Tensor = torch.cat(distances_list, dim=1)
        if self.cluster_use_random_sort:
            sort_random_index: np.array = \
                np.random.randint(0, self.eps_random_tensor.shape[0], each_envs_distances.shape[0])
            # random_eps可能代表随机扰动
            random_eps: torch.Tensor = self.eps_random_tensor[sort_random_index]
            each_envs_distances = each_envs_distances + random_eps
        new_envs: torch.Tensor = torch.argmin(each_envs_distances, dim=1)

        return new_envs

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor,
                batch_pop_envs_tensor, batch_con_envs_tensor, batch_env_classifier, batch_sample_weights
        )) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor,
                                        self.scores_tensor, self.pop_envs, self.con_envs,
                                        self.env_classifier, self.sample_weights)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                batch_pop_envs_tensor=batch_pop_envs_tensor,
                batch_con_envs_tensor=batch_con_envs_tensor,
                batch_env_classifier=batch_env_classifier,
                batch_sample_weights=batch_sample_weights,
                alpha=self.alpha,
                batch_index=batch_index
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def cluster(self) -> int:
        """
        """
        self.model.eval()

        new_env_tensors_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor, self.scores_tensor)):
            new_env_tensor: torch.Tensor = self.cluster_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor
            )

            new_env_tensors_list.append(new_env_tensor)

        all_new_env_tensors: torch.Tensor = torch.cat(new_env_tensors_list, dim=0)

        envs_diff: torch.Tensor = (self.envs - all_new_env_tensors) != 0
        diff_num: int = int(torch.sum(envs_diff))
        self.envs = all_new_env_tensors
        return diff_num

    def update_each_env_count(self):
        result_dict: dict = {}
        for env in range(self.envs_num):
            cnt = torch.sum(self.envs == env)
            result_dict[env] = cnt
        self.each_env_count.update(result_dict)

    def stat_envs(self) -> dict:
        """
        @return:
        """
        result: dict = dict()
        class_rate_np: np.array = np.zeros(self.envs_num)
        for env in range(self.envs_num):
            cnt: int = int(torch.sum(self.envs == env))
            result[env] = cnt
            class_rate_np[env] = min(cnt + 1, self.scores_tensor.shape[0] - 1)

        class_rate_np = class_rate_np / self.scores_tensor.shape[0]
        self.class_weights = torch.Tensor(class_rate_np).to(self.device)
        self.sample_weights = self.class_weights[self.envs]

        return result

    def train(self, silent: bool = False, auto: bool = False):
        """

        @param silent:
        @param auto:
        @return:
        """
        print(Fore.GREEN)
        print('=' * 30, 'train started!!!', '=' * 30)
        print(Style.RESET_ALL)

        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        max_metric, max_same_cnt = 0, 0

        # self.stat_envs()

        if not silent and not auto:
            print(Fore.BLUE)
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            # Training data, in which there is model training information
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)

            if not silent and not auto:
                print(Fore.GREEN)
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                max_same_cnt += 1

                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print(Fore.BLUE)
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

                if max_metric < temp_eval_result['recall'][50]:
                    max_metric = temp_eval_result['recall'][50]
                    max_same_cnt = 0

            if max_same_cnt >= 10:
                break

        print('=' * 30, 'train finished!!!', '=' * 30)
        return (loss_result_list, train_epoch_index_list), \
            (test_result_list, test_epoch_list), \
            (cluster_diff_num_list, envs_cnt_list, cluster_epoch_list)





