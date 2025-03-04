import pickle
import sys
import torch

from dataloader import BaseImplicitDataLoader, BaseExplicitDataLoader
from models import GeneralDebiasExplicitRecommender
import numpy as np
# from train import ImplicitTrainManager

from utils import eval_mini_batch, mini_batch, merge_dict
from torch import nn


def get_label(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def recall_precision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = np.array([k for i in range(len(test_data))])
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred / precis_n)

    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


class ImplicitTestManager:

    def __init__(self,
                 model: GeneralDebiasExplicitRecommender,
                 data_loader: BaseImplicitDataLoader,
                 test_tensor_data: dict,
                 test_batch_size: int,
                 top_k_list: list,
                 use_item_pool: bool = False):
        self.model: GeneralDebiasExplicitRecommender = model
        self.data_loader: BaseImplicitDataLoader = data_loader
        self.test_tensor_data: dict = test_tensor_data

        self.batch_size: int = test_batch_size
        self.top_k_list: list = top_k_list
        self.top_k_list.sort(reverse=False)
        self.use_item_pool: bool = use_item_pool

    def evaluate_batch(self, batch_users_tensor: torch.Tensor,
                       batch_users_list: list,
                       batch_purchase_tensor: torch.Tensor,
                       batch_users_ground_truth) -> dict:
        """
        :param batch_users_tensor: content must be same with 'batch_users_list'
        :param batch_users_list: content must be same with 'batch_users_tensor'
        :param batch_users_ground_truth: list of sets
        :return result: metrics dict
        """
        if torch.cuda.is_available():
            batch_purchase_tensor = batch_purchase_tensor.cuda()
        rating_matrix: torch.Tensor = self.model.predict(batch_users_tensor, batch_purchase_tensor)

        mask_users, mask_items = [], []
        """
        这一步也许可以优化，如果test时间过长，就优化一波
        """
        for idx, user_id in enumerate(batch_users_list):
            mask_items_set = self.data_loader.user_mask_items(user_id)
            mask_users += [idx] * len(mask_items_set)
            mask_items += list(mask_items_set)
        rating_matrix[mask_users, mask_items] = -(1 << 10)

        if self.use_item_pool:
            high_light_users, high_light_items = [], []
            """
            这一步也许可以优化，如果test时间过长，就优化一波
            """
            for idx, user_id in enumerate(batch_users_list):
                high_light_items_set = self.data_loader.user_highlight_items(
                    user_id)
                high_light_users += [idx] * len(high_light_items_set)
                high_light_items += list(high_light_items_set)
            rating_matrix[high_light_users, high_light_items] = -(1 << 10)

        # print(rating_matrix.shape)
        _, predict_items = torch.topk(rating_matrix, k=max(self.top_k_list))
        predict_items: np.array = predict_items.cpu().numpy()
        # print(predict_items)
        # print(type(predict_items))
        # print(type(predict_items[0]))
        predict_items_list: list = predict_items.tolist()

        r = get_label(batch_users_ground_truth, predict_items_list)

        pre, recall, ndcg = [], [], []
        for k in self.top_k_list:
            ret = recall_precision_ATk(batch_users_ground_truth, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ret = NDCGatK_r(batch_users_ground_truth, r, k)
            ndcg.append(ret)

        result_dict: dict = {
            'ndcg': ndcg,
            'recall': recall,
            'precision': pre,
        }

        return result_dict

    def evaluate(self) -> dict:
        self.model.eval()

        def _merge_dicts_elements_func(elements_list, **args):
            user_num: int = args['user_num']
            return (np.sum(np.array(elements_list), axis=0) /
                    float(user_num)).tolist()

        # load data of test set 
        all_test_user_tensor: torch.Tensor = self.data_loader.all_test_users_by_sorted_tensor
        all_test_user_list: list = self.data_loader.all_test_users_by_sorted_list
        all_test_ground_truth: list = self.data_loader.get_sorted_all_test_users_ground_truth

        result_dicts_list: list = []

        for (batch_index,
             ((batch_users_tensor, batch_users_list, batch_users_ground_truth), batch_purchase_tensor)) in enumerate(
            eval_mini_batch(self.batch_size,
                            self.data_loader.user_positive_interaction,
                            self.data_loader.item_num,
                            all_test_user_tensor,
                            all_test_user_list,
                            all_test_ground_truth)):
            batch_result: dict = self.evaluate_batch(
                batch_users_tensor=batch_users_tensor,
                batch_users_list=batch_users_list,
                batch_purchase_tensor=batch_purchase_tensor,
                batch_users_ground_truth=batch_users_ground_truth)

            result_dicts_list.append(batch_result)
        # print(result_dicts_list)
        result_dict: dict = merge_dict(dict_list=result_dicts_list,
                                       merge_func=_merge_dicts_elements_func,
                                       user_num=len(all_test_user_list))

        reformat_result_dict: dict = {}
        for metric in result_dict.keys():
            values: list = result_dict[metric]
            metric_result_dict: dict = {}
            for idx, value in enumerate(values):
                metric_result_dict[self.top_k_list[idx]] = values[idx]
            reformat_result_dict[metric] = metric_result_dict

        return reformat_result_dict


class ExplicitTestManager:

    def __init__(
            self,
            model: GeneralDebiasExplicitRecommender,
            data_loader: BaseExplicitDataLoader,
            test_tensor_data: dict,
            test_batch_size: int,
            top_k_list: list,
            use_item_pool: bool = False):
        self.model: GeneralDebiasExplicitRecommender = model
        self.data_loader: BaseExplicitDataLoader = data_loader
        self.test_tensor_data: dict = test_tensor_data

        self.batch_size: int = test_batch_size
        self.top_k_list: list = top_k_list
        self.top_k_list.sort(reverse=False)
        self.use_item_pool: bool = use_item_pool

    def evaluate_batch(self, batch_users_tensor: torch.Tensor,
                       batch_users_list: list,
                       batch_users_ground_truth) -> dict:
        """
        :param batch_users_tensor: content must be same with 'batch_users_list'
        :param batch_users_list: content must be same with 'batch_users_tensor'
        :param batch_users_ground_truth: list of sets
        :return result: metrics dict
        """

        # batch_purchase_tensor = batch_purchase_tensor.cuda()
        rating_matrix: torch.Tensor = self.model.predict(batch_users_tensor)

        mask_users, mask_items = [], []

        for idx, user_id in enumerate(batch_users_list):
            mask_items_set = self.data_loader.user_mask_items(user_id)
            mask_users += [idx] * len(mask_items_set)
            mask_items += list(mask_items_set)
        rating_matrix[mask_users, mask_items] = -(1 << 10)

        if self.use_item_pool:
            high_light_users, high_light_items = [], []

            for idx, user_id in enumerate(batch_users_list):
                high_light_items_set = self.data_loader.user_highlight_items(
                    user_id)
                high_light_users += [idx] * len(high_light_items_set)
                high_light_items += list(high_light_items_set)
            rating_matrix[high_light_users, high_light_items] = -(1 << 10)

        # print(rating_matrix.shape)
        _, predict_items = torch.topk(rating_matrix, k=max(self.top_k_list))
        predict_items: np.array = predict_items.cpu().numpy()
        # print(predict_items)
        # print(type(predict_items))
        # print(type(predict_items[0]))
        predict_items_list: list = predict_items.tolist()

        r = get_label(batch_users_ground_truth, predict_items_list)

        pre, recall, ndcg = [], [], []
        for k in self.top_k_list:
            ret = recall_precision_ATk(batch_users_ground_truth, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ret = NDCGatK_r(batch_users_ground_truth, r, k)
            ndcg.append(ret)

        result_dict: dict = {
            'ndcg': ndcg,
            'recall': recall,
            'precision': pre,
        }

        return result_dict

    def evaluate(self) -> dict:
        self.model.eval()

        def _merge_dicts_elements_func(elements_list, **args):
            user_num: int = args['user_num']
            return (np.sum(np.array(elements_list), axis=0) /
                    float(user_num)).tolist()

        # load data of test set 
        all_test_user_tensor: torch.Tensor = self.data_loader.all_test_users_by_sorted_tensor
        all_test_user_list: list = self.data_loader.all_test_users_by_sorted_list
        all_test_ground_truth: list = self.data_loader.get_sorted_all_test_users_ground_truth

        result_dicts_list: list = []

        for (batch_index, (batch_users_tensor, batch_users_list, batch_users_ground_truth)) in \
                enumerate(mini_batch(self.batch_size, all_test_user_tensor, all_test_user_list, all_test_ground_truth)):
            batch_result: dict = self.evaluate_batch(
                batch_users_tensor=batch_users_tensor,
                batch_users_list=batch_users_list,
                batch_users_ground_truth=batch_users_ground_truth)

            result_dicts_list.append(batch_result)
        # print(result_dicts_list)
        result_dict: dict = merge_dict(dict_list=result_dicts_list,
                                       merge_func=_merge_dicts_elements_func,
                                       user_num=len(all_test_user_list))

        reformat_result_dict: dict = {}
        for metric in result_dict.keys():
            values: list = result_dict[metric]
            metric_result_dict: dict = {}
            for idx, value in enumerate(values):
                metric_result_dict[self.top_k_list[idx]] = values[idx]
            reformat_result_dict[metric] = metric_result_dict

        return reformat_result_dict



