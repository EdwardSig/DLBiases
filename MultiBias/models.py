import torch
from torch import nn


class EnvClassifier(nn.Module):

    def __init__(self):
        super(EnvClassifier, self).__init__()

    def forward(self, invariant_preferences):
        raise NotImplementedError

    def get_L2_reg(self) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self) -> torch.Tensor:
        raise NotImplementedError


class BasicRecommender(nn.Module):

    def __init__(self, user_num: int, item_num: int):
        super(BasicRecommender, self).__init__()
        self.user_num: int = user_num
        self.item_num: int = item_num

    def forward(self, *args):
        raise NotImplementedError

    def get_L2_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, user_id) -> torch.Tensor:
        raise NotImplementedError


class BasicExplicitRecommender(nn.Module):

    def __init__(self, user_num: int, item_num: int):
        super(BasicExplicitRecommender, self).__init__()
        self.user_num: int = user_num
        self.item_num: int = item_num

    def forward(self, *args):
        raise NotImplementedError

    def get_L2_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, user_id, item_id) -> torch.Tensor:
        raise NotImplementedError


class LinearLogSoftMaxEnvClassifier(EnvClassifier):

    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        # self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        # result = self.classifier_func(result)
        return result

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight,
                          1) / self.elements_num + torch.norm(
                              self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight,
                          2).pow(2) / self.elements_num + torch.norm(
                              self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class GeneralDebiasImplicitRecommender(BasicRecommender):

    def __init__(self, user_num: int, item_num: int, env_num: int):
        super(GeneralDebiasImplicitRecommender,
              self).__init__(user_num=user_num, item_num=item_num)
        self.env_num: int = env_num

    def forward(self, *args):
        raise NotImplementedError

    def get_L2_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, user_id) -> torch.Tensor:
        raise NotImplementedError

    def cluster_predict(self, *args) -> torch.Tensor:
        raise NotImplementedError


class GeneralDebiasExplicitRecommender(BasicExplicitRecommender):

    def __init__(self, user_num: int, item_num: int, env_num: int):
        super(GeneralDebiasExplicitRecommender,
              self).__init__(user_num=user_num, item_num=item_num)
        self.env_num: int = env_num

    def forward(self, *args):
        raise NotImplementedError

    def get_L2_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, user_id, item_id) -> torch.Tensor:
        raise NotImplementedError

    def cluster_predict(self, *args) -> torch.Tensor:
        raise NotImplementedError


class BiasFactMFExplicit(GeneralDebiasExplicitRecommender):

    def __init__(self,
                 user_num: int,
                 item_num: int,
                 pop_env_num: int,
                 con_env_num: int,
                 factor_num: int,
                 drop_rate: float,
                 hidden_layers: list,
                 reg_only_embed: bool = False,
                 reg_env_embed: bool = True):
        super(BiasFactMFExplicit, self).__init__(user_num=user_num,
                                                   item_num=item_num,
                                                   env_num=pop_env_num+con_env_num)
        self.factor_num: int = factor_num

        # MF
        self.user_embedding = nn.Embedding(user_num, factor_num)
        self.item_embedding = nn.Embedding(item_num, factor_num)

        # Bias environment identification implicit vector
        self.embed_user_env_aware = nn.Embedding(user_num, factor_num)
        self.embed_item_env_aware = nn.Embedding(item_num, factor_num)
        self.pop_env2hidden = nn.Embedding(pop_env_num, factor_num)
        self.con_env2hidden = nn.Embedding(con_env_num, factor_num)

        # Bias classifier
        self.env_classifier: EnvClassifier = LinearLogSoftMaxEnvClassifier(
            factor_num, pop_env_num+con_env_num)
        self.output_func = nn.ReLU()

        self.reg_only_embed: bool = reg_only_embed

        self.reg_env_embed: bool = reg_env_embed

        self._init_weight()

    def _init_weight(self):
        # Initialize parameter
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std=0.01)
        nn.init.normal_(self.pop_env2hidden.weight, std=0.01)
        nn.init.normal_(self.con_env2hidden.weight, std=0.01)

    def forward(self, users_id, items_id, pop_envs_id, con_envs_id, alpha=0):

        ###############################################
        # MF
        ###############################################
        user_emb = self.user_embedding(users_id)
        item_emb = self.item_embedding(items_id)
        pop_envs_embed: torch.Tensor = self.pop_env2hidden(pop_envs_id)
        con_envs_embed: torch.Tensor = self.con_env2hidden(con_envs_id)
        envs_embed: torch.Tensor = pop_envs_embed * con_envs_embed

        mf_score = self.output_func(torch.sum(user_emb * item_emb * envs_embed, dim=1))

        ###############################################
        # Bias factor identification
        ###############################################
        users_embed_env_aware: torch.Tensor = self.embed_user_env_aware(
            users_id)
        items_embed_env_aware: torch.Tensor = self.embed_item_env_aware(
            items_id)

        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed

        env_aware_score: torch.Tensor = self.output_func(
            torch.sum(env_aware_preferences, dim=1))
        env_outputs: torch.Tensor = self.env_classifier(env_aware_preferences)

        # env_aware_score, env_outputs = self.env_classifier(env_aware_preferences)
        # env_outputs = self.output_func(env_outputs)

        return mf_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)

    def get_users_reg(self, users_id, norm: int):
        env_aware_embed_gmf: torch.Tensor = self.embed_user_env_aware(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = (env_aware_embed_gmf.norm(2).pow(2)) / (
                float(len(users_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = (env_aware_embed_gmf.norm(1)) / (
                float(len(users_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        env_aware_embed_gmf: torch.Tensor = self.embed_item_env_aware(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = (env_aware_embed_gmf.norm(2).pow(2)) / (
                    float(len(items_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = (env_aware_embed_gmf.norm(1)) / (
                    float(len(items_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_pop_envs_reg(self, envs_id, norm: int):
        embed_gmf: torch.Tensor = self.pop_env2hidden(envs_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (
                    float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (
                    float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_con_envs_reg(self, envs_id, norm: int):
        embed_gmf: torch.Tensor = self.con_env2hidden(envs_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (
                    float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (
                    float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L2_reg(self, users_id, items_id, pop_envs_id, con_envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L2_reg()
            result = result + (self.get_users_reg(users_id, 2) +
                               self.get_items_reg(items_id, 2))
        else:
            result = self.get_users_reg(users_id, 2) + self.get_items_reg(
                items_id, 2)

        if self.reg_env_embed:
            result = result + self.get_pop_envs_reg(pop_envs_id, 2) + self.get_con_envs_reg(con_envs_id, 2)
        # print('L2', result)
        return result

    def get_L1_reg(self, users_id, items_id, pop_envs_id, con_envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L1_reg()
            result = result + (self.get_users_reg(users_id, 1) +
                               self.get_items_reg(items_id, 1))
        else:
            result = self.get_users_reg(users_id, 1) + self.get_items_reg(
                items_id, 1)
        if self.reg_env_embed:
            result = result + self.get_pop_envs_reg(pop_envs_id, 1) + self.get_con_envs_reg(con_envs_id, 1)
        # print('L2', result)
        return result

    def predict(self, users_id, item_id=None):

        user_emb = self.user_embedding(users_id)
        # item_emb = self.item_embedding(predict_items_id)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_emb, all_item_e.transpose(0, 1)).view(-1)

        return score.reshape(len(users_id), self.item_num)



