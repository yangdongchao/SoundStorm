# ------------------------------------------
# Diffsound
# written by Dongchao Yang
# code based on https://github.com/cientgu/VQ-Diffusion
# ------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
from soundstorm.s2.utils.misc import instantiate_from_config
from torch import nn
from torch.cuda.amp import autocast
# from torchmetrics.classification import MulticlassAccuracy
eps = 1e-8


# 对 num_dims 后面的维度进行求和
def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


# log(1-e_a)
def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


# M + log(e_(a-M)+e_(b-M))
def log_add_exp(a, b):
    # e(-70) 近似为0 
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# x_shape torch.Size([2, 2888, 1024])
def extract(a, t, x_shape):
    # b,剩下的
    b, *_ = t.shape
    out = a.gather(-1, t)
    # (b,1,1)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    # 根据数值产生 one-hot 向量, [2, 1024, 2888]
    x_onehot = F.one_hot(x, num_classes)
    # 0,-1,1
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # [2, 2888, 1024]
    x_onehot = x_onehot.permute(permute_order)
    # 对 one-hot 取 log? [2, 2888, 1024]
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


# 根据 log_onehot 向量，找到对应的 index
def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def alpha_schedule(time_step,
                   N=100,
                   att_1=0.99999,
                   att_T=0.000009,
                   ctt_1=0.000009,
                   ctt_T=0.9):
    # mask and uniform
    # it means alpha, 等差数列
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    # add 1 on the first
    att = np.concatenate(([1], att))
    # 得到从当前步到下一步乘的系数
    at = att[1:] / att[:-1]
    # denotes gama,the prob for mask token
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    # 与 att 反过来
    ctt = np.concatenate(([0], ctt))
    # reverse
    one_minus_ctt = 1 - ctt
    # 9.99991000e-01, 9.89899091e-01
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    # 9.00000000e-06, 1.01009091e-02
    ct = 1 - one_minus_ct
    # it means beta
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


class DiffusionTransformer(nn.Module):
    def __init__(
            self,
            transformer_config=None,
            diffusion_step: int=100,
            n_q: int=12,
            alpha_init_type: str='cos',
            auxiliary_loss_weight: int=0,
            adaptive_auxiliary_loss: bool=False,
            mask_weight=[1, 1], ):
        super().__init__()
        # 在 transformer_conf 文件中，加入这两个参数
        transformer_config['diffusion_step'] = diffusion_step
        self.n_q = n_q
        # 加载 transformer
        transformer_config['params']['n_q'] = self.n_q
        self.transformer = instantiate_from_config(transformer_config)
        # 1024  # 32 x 32
        # self.content_seq_len = transformer_config['content_seq_len'] 
        self.amp = False
        # 2888 #? 2887 + 1
        self.num_classes = self.transformer.content_emb.num_embed
        self.loss_type = 'vb_stochastic'
        # 迭代的次数
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        # Reparameterization trick?
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        # [1,1] , what the means? --> the loss weight on mask region and non-mask region
        self.mask_weight = mask_weight

        if alpha_init_type == "alpha1":
            at, bt, ct, att, btt, ctt = alpha_schedule(
                self.num_timesteps, N=self.num_classes)
        else:
            print("alpha_init_type is Wrong !! ")
        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)  # 对系数求log 
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)
        # log(1-e_a), log(1-ct)
        log_1_min_ct = log_1_min_a(log_ct)
        # log(1-ctt)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)
        # M + log(e_(a-M)+e_(b-M))
        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct,
                           log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct',
                             log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        self.zero_vector = None

        # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.prior_rule = 2
        # max number to sample per step
        self.prior_ps = 1300
        # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion
        self.prior_weight = 2

        self.update_n_sample(total_num=1300)

        # too slow
        # self.metric_top10 = MulticlassAccuracy(
        #     self.num_classes, top_k=10, average="micro",multidim_average="global",)
        # self.metric_top1 = MulticlassAccuracy(
        #     self.num_classes, top_k=1, average="micro",multidim_average="global",)

    def update_n_sample(self, total_num):
        # 设定每步要更新的 mask sample
        if total_num < self.num_timesteps + 1:
            self.n_sample = [0] * (self.num_timesteps - total_num
                                   ) + [1] * total_num
        else:
            avg = total_num // (self.num_timesteps - 2)
            add = total_num - avg * (self.num_timesteps - 2) - 1
            if add > 5:
                self.n_sample = [1, 5] + [avg] * (
                    self.num_timesteps - 3
                ) + [total_num - 6 - avg * (self.num_timesteps - 3)]
            else:
                self.n_sample = [1, add] + [avg] * (self.num_timesteps - 2)

    # compute KL loss on log_prob
    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    # q(xt|xt_1)
    def q_pred_one_timestep(self, log_x_t, t):
        # at
        log_at = extract(self.log_at, t, log_x_t.shape)
        # bt
        log_bt = extract(self.log_bt, t, log_x_t.shape)
        # ct 
        log_ct = extract(self.log_ct, t, log_x_t.shape)
        # 1-ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)
        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1)
        return log_probs

    # q(xt|x0)
    def q_pred(self, log_x_start, t):
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        # at~
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)
        # bt~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)
        # ct~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)
        # 1-ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t,
                                       log_x_start.shape)
        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at,
                            log_cumprod_bt),
                log_add_exp(log_x_start[:, -1:, :] + log_1_min_cumprod_ct,
                            log_cumprod_ct)
            ],
            dim=1)
        return log_probs

    # p(x0|xt)
    def predict_start(self, log_x_t, cond_emb, x_mask, cond_emb_mask, t):
        # 核心是根据 x_t 推理出 x0
        # get the index label
        x_t = log_onehot_to_index(log_x_t)
        if self.amp is True:
            with autocast():
                out = self.transformer(x_t, cond_emb, x_mask, cond_emb_mask, t)
        else:
            # get logit
            out = self.transformer(x_t, cond_emb, x_mask, cond_emb_mask, t)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes - 1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        # ? (log(1e-30))?
        zero_vector = torch.zeros(batch_size, 1,
                                  log_pred.shape[2]).type_as(log_x_t) - 70
        # 最后一行代表 mask_token
        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred

    def cf_predict_start(self, log_x_t, cond_emb, x_mask, cond_emb_mask, t):
        return self.predict_start(log_x_t, cond_emb, x_mask, cond_emb_mask, t)

    # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0)*p(x0|xt))
    def q_posterior(self, log_x_start, log_x_t, t):
        # notice that log_x_t is onehot
        # log(p_theta(xt_1|xt)) = log(q(xt-1|xt,x0)) + log(p(x0|xt))
        #                       = log(p(x0|xt)) + log(q(xt|xt_1,x0)) + log(q(xt_1|x0)) - log(q(xt|x0))  (*)
        # log_x_start=log_x0_recon (the results of prediction), log_x_t=log_xt, t=t
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        # get sample
        onehot_x_t = log_onehot_to_index(log_x_t)
        # 选出为 mask token 的
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)
        # [b, 1, 1] (全0)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        # [2, 1, 1024]
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(
            -1, -1, log_x_start.shape[2])
        # log(q(xt|x0))
        # x_t 在向前 t 步, 或者说把 x_t 当成 x_0 使用？ 
        log_qt = self.q_pred(log_x_t, t)
        # 代表 mask 的位置，全设为0
        log_qt = torch.cat((log_qt[:, :-1, :], log_zero_vector), dim=1)
        #  # ct~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)
        # log_x_start=log_x0_recon, b,1,1
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        ct_cumprod_vector = torch.cat(
            (ct_cumprod_vector, log_one_vector), dim=1)
        # mask token 的部分，全设为 ct_
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector
        # log(q(xt|xt_1,x0))
        # q(xt|xt_1), 只向前一步,因为用了 at, bt
        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)
        log_qt_one_timestep = torch.cat(
            (log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1)
        # ct
        log_ct = extract(self.log_ct, t, log_x_start.shape)
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector
        # log(p(x0|xt)/q(xt|x0))
        q = log_x_start - log_qt
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        # norm(log(p(x0|xt)/q(xt|x0)))  to leverage self.q_pred
        q = q - q_log_sum_exp
        # get (*), last term is re-norm
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(
            q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_emb, t, x_mask, condition_mask):
        if self.parametrization == 'x0':
            log_x_recon = self.cf_predict_start(log_x, cond_emb, x_mask,
                                                condition_mask, t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            print('please choose x0 as parametrization trick')
            assert 1 == 2
            log_model_pred = self.predict_start(log_x, cond_emb, t)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(
            self,
            log_x,
            cond_emb,
            t,
            x_mask,
            condition_mask,
            sampled=None,
            to_sample=None
    ):  # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        # For the begin, sampled is all zero, to_sample is choosed based on self.n_sample
        model_log_prob, log_x_recon = self.p_pred(log_x, cond_emb, t, x_mask,
                                                  condition_mask)
        # max number to sample per step
        max_sample_per_step = self.prior_ps
        # prior_rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        if t[0] > 0 and self.prior_rule > 0 and to_sample is not None:
            # get the index
            log_x_idx = log_onehot_to_index(log_x)

            if self.prior_rule == 1:
                # B, N
                score = torch.ones(
                    (log_x.shape[0], log_x.shape[2])).to(log_x.device)
            elif self.prior_rule == 2:
                # get purity
                score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                # norm
                score /= (score.max(dim=1, keepdim=True).values + 1e-10)

            if self.prior_rule != 1 and self.prior_weight > 0:
                # probability adjust parameter, prior_weight: 'r' in Equation.11 of Improved VQ-Diffusion
                prob = ((1 + score * self.prior_weight).unsqueeze(1) *
                        log_x_recon).softmax(dim=1)
                prob = prob.log().clamp(-70, 0)
            else:
                prob = log_x_recon
            # set as inf
            prob[:, 1024:, :] = -70
            # get x^0
            out = self.log_sample_categorical(prob)
            out_idx = log_onehot_to_index(out)

            out2_idx = log_x_idx.clone()
            _score = score.clone()
            if _score.sum() < 1e-6:
                _score += 1
            # 标记不为 mask 的位置
            _score[log_x_idx != self.num_classes - 1] = 0

            for i in range(log_x.shape[0]):
                # 需要采样的
                n_sample = min(to_sample - sampled[i], max_sample_per_step)
                if to_sample - sampled[i] - n_sample == 1:
                    n_sample = to_sample - sampled[i]
                if n_sample <= 0:
                    continue
                # 根据权重采样，结果为 index
                sel = torch.multinomial(_score[i], n_sample)
                out2_idx[i][sel] = out_idx[i][sel]
                sampled[i] += (
                    (out2_idx[i] != self.num_classes - 1).sum() -
                    (log_x_idx[i] != self.num_classes - 1).sum()).item()

            out = index_to_log_onehot(out2_idx, self.num_classes)
        else:
            # Gumbel sample
            out = self.log_sample_categorical(model_log_prob)
            # we change the 1024  to self.prior_ps
            sampled = [self.prior_ps] * log_x.shape[0]

        if to_sample is not None:
            return out, sampled
        else:
            return out

    # use gumbel to sample onehot vector from log probability
    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        # 产生一定的噪声
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)

        # 每行最大值所在的 index
        sample = (gumbel_noise + logits).argmax(dim=1)
        # 又把 index 转为 log one-hot
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    # diffusion step, q(xt|x0) and sample xt
    def q_sample(self, log_x_start, t):
        # 从 x_0 开始，往前走 t 步 (马尔科夫链),获得 logq(xt|x0)
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        # 根据概率分布，进行采样
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            # 当矩阵里每个值都大于 10 时，才不使用 uniform 采样
            if not (self.Lt_count > 10).all():
                # print('use uniform... ')
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            # Overwrite decoder term with L1.
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            # 采 index 权重大的，采到的几率就越大
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            # input 张量可以看成一个权重张量，每一个元素代表其在该行中的权重。如果有元素为0，那么在其他不为0的元素被取干净之前，这个元素是不会被取到的。
            # 根据 index, 找到对应的值
            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            # 从 [0, num_timesteps] 随机产生 b 个数
            t = torch.randint(
                0, self.num_timesteps, (b, ), device=device).long()
            # 概率一直都是0.01?
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def topk_accuracy(self, predictions, targets, k: int=1, mask=None):
        _, indices = torch.topk(predictions, k, dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(indices)
        if mask is not None:
            # Apply mask to the targets and indices
            mask_expanded = mask.unsqueeze(1).expand_as(indices)
            targets_expanded = targets_expanded[mask_expanded]
            indices = indices[mask_expanded]

        correct = torch.sum(indices == targets_expanded)
        accuracy = torch.sum(correct) / targets.numel()
        return accuracy

    def _train_loss(
            self,
            x,
            cond_emb,
            x_mask=None,
            cond_emb_mask=None,
            # get the KL loss
            is_train: bool=True):
        b, device = x.size(0), x.device
        assert self.loss_type == 'vb_stochastic'
        # (b, N)
        x_start = x
        # 时间采样
        t, pt = self.sample_time(b, device, 'importance')
        # 将数值代表，转换为由 one-hot 向量组成的矩阵, 其中每个向量最大值所在的索引就是原始的值
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        # 通过采样获得 log_xt, 随机采得
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        # get b, N
        xt = log_onehot_to_index(log_xt)
        ############### go to p_theta function ###############
        # P_theta(x0|xt) shape (B, num_classes, T)
        log_x0_recon = self.predict_start(
            log_xt, cond_emb, x_mask, cond_emb_mask, t=t)
        # go through q(xt_1|xt,x0)
        log_model_prob = self.q_posterior(
            log_x_start=log_x0_recon, log_x_t=log_xt, t=t)
        ################## compute acc list ################
        # 获得预测的 x_0, shape (B, T)
        x0_recon = log_onehot_to_index(log_x0_recon)
        # 真实值, shape (B, T)
        x0_real = x_start
        # 直接采样 x_(t-1), 与 x(t) 相同的数量
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        # (B, Len) --> (B, n_q, len)
        x_mask_repeat = x_mask.unsqueeze(1).repeat(1, self.n_q, 1)
        # B, Len*n_q
        x_mask = x_mask_repeat.reshape(x_mask_repeat.shape[0], -1)
        # 对 batch 维度进行遍历
        for index in range(t.size()[0]):
            # batch 里面的每一条有一个时间随机数
            this_t = t[index].item()
            # 获得当前样本的 mask 值
            tmp_mask = ~x_mask[index]
            # 只保留非 mask 部分
            # tmp_x0_recon 和 tmp_x0_real 长度一致, shape (T)
            tmp_x0_recon = x0_recon[index][tmp_mask]
            tmp_x0_real = x0_real[index][tmp_mask]
            same_rate = (
                tmp_x0_recon == tmp_x0_real).sum().cpu() / tmp_x0_real.shape[0]
            # 平均移动更新
            self.diffusion_acc_list[this_t] = same_rate.item(
            ) * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            tmp_xt_1_recon = xt_1_recon[index][tmp_mask]
            tmp_xt_recon = xt_recon[index][tmp_mask]
            same_rate = (tmp_xt_1_recon == tmp_xt_recon
                         ).sum().cpu() / tmp_xt_recon.shape[0]
            self.diffusion_keep_list[this_t] = same_rate.item(
            ) * 0.1 + self.diffusion_keep_list[this_t] * 0.9

        # compute log_true_prob now 
        # using true label to calculate
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        # xt 中被 mask 的区域
        mask_region = (xt == self.num_classes - 1).float()
        mask_weight = mask_region * self.mask_weight[0] + (
            1. - mask_region) * self.mask_weight[1]
        # 去掉 padding 部分
        # x_mask 为 True 的部分表示 target_acoustic 的位置
        kl = kl * mask_weight * (~x_mask)
        kl = sum_except_batch(kl)
        # 分类的概率 e_0
        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        # mask padding 部分
        decoder_nll = decoder_nll * (~x_mask)
        decoder_nll = sum_except_batch(decoder_nll)
        # t 为 0, p(x_0|x_1,y)
        mask = (t == torch.zeros_like(t)).float()
        # t 为 0 时，直接计算分类损失,否则计算kl
        kl_loss = mask * decoder_nll + (1. - mask) * kl

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        # 记录下 kl
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        # 记录加的次数, 也可理解为选择了时间步 t 的次数
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        # pt 代表得到采样时间的概率
        loss1 = kl_loss / pt
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train is True:
            kl_aux = self.multinomial_kl(log_x_start[:, :-1, :],
                                         log_x0_recon[:, :-1, :])
            kl_aux = kl_aux * mask_weight * (~x_mask)
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss is True:
                addition_loss_weight = t / self.num_timesteps + 1.0
            else:
                addition_loss_weight = 1.0
            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        # calculate acc in cpu cause it cost to much gpu when use MulticlassAccuracy
        # self.metric_top1.to('cpu')
        # self.metric_top10.to('cpu')
        probs = log_model_prob.cpu()
        targets = x_start.cpu()
        x_mask_cpu_reverse = ~x_mask.cpu()

        # top1_acc = self.metric_top1(probs, targets)
        top1_acc = self.topk_accuracy(
            probs, targets, k=1, mask=x_mask_cpu_reverse)
        # top10_acc = self.metric_top10(probs, targets)
        top10_acc = self.topk_accuracy(
            probs, targets, k=10, mask=x_mask_cpu_reverse)

        top1_acc = top1_acc.to(vb_loss.device)
        top10_acc = top10_acc.to(vb_loss.device)

        return log_model_prob, vb_loss, top1_acc, top10_acc

    @property
    def device(self):
        return self.transformer.content_emb.pos_emb.weight.device

    def parameters(self, recurse: bool=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    # full param name
                    fpn = '%s.%s' % (mn, pn) if mn else pn

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(
                            m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(
                            m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = [
                'pos_emb', 'width_emb', 'height_emb', 'pad_emb',
                'token_type_emb'
            ]
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(
                                    getattr(getattr(self, mn), pn),
                                    torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            # if p.requires_grad}
            param_dict = {
                pn: p
                for pn, p in self.transformer.named_parameters()
            }
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(
                inter_params
            ) == 0, "parameters %s made it into both decay/no_decay sets!" % (
                str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )
            # create the pytorch optimizer object
            optim_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(list(decay))],
                    "weight_decay": 0.01
                },
                {
                    "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                    "weight_decay": 0.0
                },
            ]
            return optim_groups

    def forward(self,
                input,
                return_loss: bool=True,
                return_logits: bool=True,
                is_train: bool=True,
                **kwargs):
        # {'condition_text_embed': cond, 'condition_mel_mask': cond_mask, 'content_token': }
        # the input is prompt_semantic, target_semantic, prompt_acoustic, target_acoustic
        if kwargs.get('autocast') is True:
            self.amp = True
        batch_size = input['target_acoustics'].shape[0]
        device = input['target_acoustics'].device
        # 1) get embeddding for condition and content prepare input
        content_token = input['target_acoustics'].reshape(batch_size, -1)
        # 目前先不使用 mask, 因为我们直接 padding eos
        content_token_mask = None
        content_token_mask = input['x_mask']
        # cont_emb = self.content_emb(sample_image)
        cond_emb = {}
        cond_emb_mask = None  # condition mask
        cond_emb['prompt_semantics'] = input['prompt_semantics']
        cond_emb['prompt_acoustics'] = input['prompt_acoustics']
        cond_emb['target_semantics'] = input['target_semantics']

        if is_train is True:
            log_model_prob, loss, top1_acc, top10_acc = self._train_loss(
                content_token, cond_emb, content_token_mask, cond_emb_mask)
            # ? mask
            loss = loss.sum() / (content_token.size()[0] *
                                 content_token.size()[1])
        # 4) get output, especially loss
        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)
        if return_loss:
            out['loss'] = loss
            out['top1_acc'] = top1_acc
            out['top10_acc'] = top10_acc
        self.amp = False
        return out

    def sample(self, batch, filter_ratio: float=0.5, return_logits: bool=False):
        real_content = batch['target_acoustics']
        batch_size = real_content.shape[0]
        device = self.log_at.device
        # 100*filter_ratio
        start_step = int(self.num_timesteps * filter_ratio)
        condition = {}
        condition['prompt_semantics'] = batch['prompt_semantics']
        condition['prompt_acoustics'] = batch['prompt_acoustics']
        condition['target_semantics'] = batch['target_semantics']
        content_token_mask = batch['x_mask']
        condition_mask = None
        # get cont_emb and cond_emb
        # when filter_ratio==0
        if start_step == 0:
            # use full mask sample
            # Note that this part only support mask, mask and uniform strategies, if you use uniform strategy
            predict_len = batch['target_acoustics'].shape[-1] * self.n_q
            # unpdate prior_ps and n_sample
            self.prior_ps = predict_len
            self.update_n_sample(predict_len)
            # [b, 256, 265]
            zero_logits = torch.zeros(
                (batch_size, self.num_classes - 1, predict_len), device=device)
            # [b, 1, 265]
            one_logits = torch.ones((batch_size, 1, predict_len), device=device)
            # 每个 token 全是 mask
            mask_logits = torch.cat((zero_logits, one_logits), dim=1)
            log_z = torch.log(mask_logits)
            start_step = self.num_timesteps
            with torch.no_grad():
                # 99,0
                for diffusion_index in range(start_step - 1, -1, -1):
                    # 用 diffusion_index 填充，shape=(b,)
                    t = torch.full(
                        (batch_size, ),
                        diffusion_index,
                        device=device,
                        dtype=torch.long)
                    # 初始时 sampled 全设为 0
                    sampled = [0] * log_z.shape[0]
                    while min(sampled) < self.n_sample[diffusion_index]:
                        # log_z is log_onehot
                        log_z, sampled = self.p_sample(
                            log_z, condition, t, content_token_mask,
                            condition_mask, sampled,
                            self.n_sample[diffusion_index])
        else:
            print('error, we must sample from zero')
        # transfer from one-hot to index
        content_token = log_onehot_to_index(log_z)
        # return the predict content_token
        output = {'pre_content_token': content_token}
        # false
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output
