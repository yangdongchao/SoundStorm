# ------------------------------------------
# Diffsound
# written by Dongchao Yang
# code based on https://github.com/cientgu/VQ-Diffusion
# ------------------------------------------

import math
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from hydra.utils import instantiate
from sound_synthesis2.utils.misc import instantiate_from_config
from torch.cuda.amp import autocast
eps = 1e-8

def sum_except_batch(x, num_dims=1): # 对num_dims 后面的维度进行求和
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a): # log(1-e_a)
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b): # M + log(e_(a-M)+e_(b-M))
    maximum = torch.max(a, b) # e(-70) 近似为0
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape): # x_shape torch.Size([2, 2888, 1024])
    # print('a ',a)
    # print('t ',t)
    # print('x_shape ',x_shape)
    b, *_ = t.shape # b,剩下的
    out = a.gather(-1, t) # 
    # print('out ',out)
    # print('(len(x_shape) - 1)) ',(len(x_shape) - 1))
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # (b,1,1)

def log_categorical(log_x_start, log_prob): # ?
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes): # 
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes) # 根据数值产生one-hot向量,[2, 1024, 2888]
    # print('x_onehot ', x_onehot.shape)
    permute_order = (0, -1) + tuple(range(1, len(x.size()))) # 0,-1,1
    # print('permute_order ',permute_order)
    x_onehot = x_onehot.permute(permute_order) # [2, 2888, 1024]
    # print('x_onehot ', x_onehot.shape)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30)) # 对one-hot 取log? [2, 2888, 1024]
    # print('log_x ',log_x.shape)
    return log_x

def log_onehot_to_index(log_x): # 根据log_onehot向量，找到对应的index
    return log_x.argmax(1)

def alpha_schedule_mask_only(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1 # it means alpha, 等差数列
    # print('att ',att.shape)
    # print('att ',att)
    att = np.concatenate(([1], att)) # add 1 on the first
    # print('att1 ',att)
    # assert 1==2
    at = att[1:]/att[:-1] # 得到从当前步到下一步乘的系数
    # print('at ',at.shape)
    # print('at ',at)
    # assert 1==2
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1 # denotes gama,the prob for mask token
    # print('ctt ',ctt) # 与att反过来
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt # reverse
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1] # 9.99991000e-01, 9.89899091e-01
    # print('one_minus_ct ',one_minus_ct)
    ct = 1-one_minus_ct # 9.00000000e-06, 1.01009091e-02
    # print('ct ',ct)
    # assert 1==2
    bt = (1-at-ct)/N # it means beta
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    # print('att ',att)
    # print('btt ',btt)
    # print('ctt ',ctt)
    # assert 1==2
    return at, bt, ct, att, btt, ctt

def alpha_schedule_uniform_only(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.1):
    # set ctt_T = ? to control
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1 # it means alpha, 等差数列
    # print('att ',att.shape)
    # print('att ',att)
    att = np.concatenate(([1], att)) # add 1 on the first
    # print('att1 ',att)
    # assert 1==2
    at = att[1:]/att[:-1] # 得到从当前步到下一步乘的系数
    # print('at ',at.shape)
    # print('at ',at)
    # assert 1==2
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1 # denotes gama,the prob for mask token
    # print('ctt ',ctt) # 与att反过来
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt # reverse
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1] # 9.99991000e-01, 9.89899091e-01
    # print('one_minus_ct ',one_minus_ct)
    ct = 1-one_minus_ct # 9.00000000e-06, 1.01009091e-02
    # print('ct ',ct)
    # assert 1==2
    bt = (1-at-ct)/N # it means beta
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    return at, bt, ct, att, btt, ctt

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.9):
    # mask and uniform
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1 # it means alpha, 等差数列
    att = np.concatenate(([1], att)) # add 1 on the first
    at = att[1:]/att[:-1] # 得到从当前步到下一步乘的系数
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1 # denotes gama,the prob for mask token
    # print('ctt ',ctt) # 与att反过来
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt # reverse
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1] # 9.99991000e-01, 9.89899091e-01
    # print('one_minus_ct ',one_minus_ct)
    ct = 1-one_minus_ct # 9.00000000e-06, 1.01009091e-02
    bt = (1-at-ct)/N # it means beta
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    return at, bt, ct, att, btt, ctt

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        condition_emb_config=None,
        transformer_config=None,
        diffusion_step=100,
        n_q = 12,
        alpha_init_type='cos',
        auxiliary_loss_weight=0,
        adaptive_auxiliary_loss=False,
        learnable_cf=False,
        mask_weight=[1,1],
    ):
        super().__init__()
        if condition_emb_config is None: # 不使用 conditional information
            self.condition_emb = None
        else:
            # for condition and config, we learn a seperate embedding
            self.condition_emb = instantiate(condition_emb_config) # 加载能获得condition embedding的模型
            #print('self.condition_emb ',self.condition_emb)
            self.condition_dim = self.condition_emb.embed_dim # 每个embedding的维度
            #print('self.condition_dim ',self.condition_dim) # self.condition_dim  512
        
        transformer_config['diffusion_step'] = diffusion_step # 在transformer_conf文件中，加入这两个参数
        self.transformer = instantiate_from_config(transformer_config) # 加载transformer
        print('transformer ', self.transformer)
        # self.content_seq_len = transformer_config['content_seq_len'] # 1024  # 32 x 32
        self.amp = False
        self.n_q = n_q
        self.num_classes = self.transformer.content_emb.num_embed # 2888 #? 2887 + 1
        #print('self.num_classes ',self.num_classes) # 2888
        self.loss_type = 'vb_stochastic'
        self.num_timesteps = diffusion_step # 迭代的次数
        self.parametrization = 'x0' # 
        self.auxiliary_loss_weight = auxiliary_loss_weight # Reparameterization trick?
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight # [1,1] , what the means? --> the loss weight on mask region and non-mask region

        if alpha_init_type == "alpha1": # ?
            at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes)
        else:
            print("alpha_init_type is Wrong !! ")
        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at) # 对系数求log 
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct) # log(1-e_a), log(1-ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct) # log(1-ctt)
        # M + log(e_(a-M)+e_(b-M))
        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

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
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        self.zero_vector = None
        if learnable_cf:
            # 设定一个可学习的空文本向量, 一个字符用一个256维度的向量表示，最大长度应该不超过2000
            self.empty_text_embed = torch.nn.Parameter(torch.randn(size=(2000, 256), requires_grad=True, dtype=torch.float64))

        self.prior_rule = 2    # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.prior_ps = 1300   # max number to sample per step
        self.prior_weight = 2  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion

        self.update_n_sample(total_num = 1300)
        
        self.learnable_cf = learnable_cf

    def update_n_sample(self, total_num):
        # 设定每步要更新的 mask sample
        if total_num < self.num_timesteps+1:
            self.n_sample = [0]*(self.num_timesteps-total_num) + [1]*total_num
        else:
            avg = total_num//(self.num_timesteps-2) # 
            add = total_num - avg * (self.num_timesteps-2) -1
            if add > 5:
                self.n_sample = [1, 5] + [avg]*(self.num_timesteps-3) + [total_num-6-avg*(self.num_timesteps-3)]
            else:
                self.n_sample = [1, add] + [avg]*(self.num_timesteps-2)

    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct
        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )
        return log_probs

    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )
        return log_probs

    def sample_samll_time(self, b, device, method='uniform'):
        # do not pass 20 diffusion steps
        t = torch.randint(1, 20, (b,), device=device).long() #从[0,num_timesteps]随机产生b个数
        return t

    def predict_start(self, log_x_t, cond_emb, x_mask, cond_emb_mask, t):          # p(x0|xt)
        # 核心是根据x_t 推理出 x0
        x_t = log_onehot_to_index(log_x_t) # get the index label
        if self.amp == True:
            with autocast():
                out = self.transformer(x_t, cond_emb, x_mask, cond_emb_mask, t)
        else:
            out = self.transformer(x_t, cond_emb, x_mask, cond_emb_mask, t) # get logit
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float() # 
        batch_size = log_x_t.size()[0]
        zero_vector = torch.zeros(batch_size, 1, log_pred.shape[2]).type_as(log_x_t)- 70 # ? (log(1e-30))?
        log_pred = torch.cat((log_pred, zero_vector), dim=1) # 最后一行代表mask_token
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred

    def cf_predict_start(self, log_x_t, cond_emb, x_mask, cond_emb_mask, t):
        return self.predict_start(log_x_t, cond_emb, x_mask, cond_emb_mask, t)

    def q_posterior(self, log_x_start, log_x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0)*p(x0|xt))
        # notice that log_x_t is onehot
        # log(p_theta(xt_1|xt)) = log(q(xt-1|xt,x0)) + log(p(x0|xt))
        #                       = log(p(x0|xt)) + log(q(xt|xt_1,x0)) + log(q(xt_1|x0)) - log(q(xt|x0))  (*)
        # log_x_start=log_x0_recon (the results of prediction), log_x_t=log_xt, t=t
        # print('log_x_start ', log_x_start.shape)
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t) # get sample
        # print('log_x_t ',log_x_t.shape)
        # print('onehot_x_t ',onehot_x_t.shape)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) #选出为mask token的
        # print('mask ', mask)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t) # b,1,1 (全0)
        # print('log_one_vector ',log_one_vector)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, log_x_start.shape[2]) #[2, 1, 1024]
        # log(q(xt|x0))
        log_qt = self.q_pred(log_x_t, t)  # x_t 在向前t步, 或者说把x_t 当成x_0使用？  
        #print('log_qt1 ',log_qt)                          
        log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1) # 代表mask的位置，全设为0
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape) #  # ct~
        #print('log_cumprod_ct ',log_cumprod_ct.shape)
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1) # log_x_start=log_x0_recon, b,1,1
        #print('ct_cumprod_vector ', ct_cumprod_vector.shape) # [2, 2887, 1]
        ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector # mask token的部分，全设为ct_
        # log(q(xt|xt_1,x0))
        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1), 只向前一步,因为用了 at,bt
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        q = log_x_start - log_qt    # log(p(x0|xt)/q(xt|x0))
        #print('q ',q.shape) # [2, 2888, 1024]
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        # print('q_log_sum_exp ',q_log_sum_exp.shape) # [2, 1, 1024]
        # assert 1==2 
        q = q - q_log_sum_exp       # norm(log(p(x0|xt)/q(xt|x0)))  to leverage self.q_pred
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp  # get (*), last term is re-norm
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_emb, t, x_mask, condition_mask): 
        if self.parametrization == 'x0':
            log_x_recon = self.cf_predict_start(log_x, cond_emb, x_mask, condition_mask, t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            print('please choose x0 as parametrization trick')
            assert 1==2
            log_model_pred = self.predict_start(log_x, cond_emb, t)
        else:
            raise ValueError
        return log_model_pred, log_x_recon
    
    @torch.no_grad()
    def p_sample(self, log_x, cond_emb, t, x_mask, condition_mask, sampled=None, to_sample=None):    # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        # For the begin, sampled is all zero, to_sample is choosed based on self.n_sample
        #print('p_sampel log_x, cond_emb',log_x.shape, cond_emb.shape )
        model_log_prob, log_x_recon = self.p_pred(log_x, cond_emb, t,  x_mask, condition_mask)
        max_sample_per_step = self.prior_ps  # max number to sample per step
        if t[0] > 0 and self.prior_rule > 0 and to_sample is not None: # prior_rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
            log_x_idx = log_onehot_to_index(log_x) # get the index

            if self.prior_rule == 1:
                score = torch.ones((log_x.shape[0], log_x.shape[2])).to(log_x.device) # B, N
            elif self.prior_rule == 2:
                score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1) # get purity
                score /= (score.max(dim=1, keepdim=True).values + 1e-10) # norm

            if self.prior_rule != 1 and self.prior_weight > 0:
                # probability adjust parameter, prior_weight: 'r' in Equation.11 of Improved VQ-Diffusion
                prob = ((1 + score * self.prior_weight).unsqueeze(1) * log_x_recon).softmax(dim=1)
                prob = prob.log().clamp(-70, 0)
            else:
                prob = log_x_recon
            #print('prob ', prob.shape)
            prob[:,1024:,:] = -70 # set as inf
            out = self.log_sample_categorical(prob) # get x^0
            out_idx = log_onehot_to_index(out)

            out2_idx = log_x_idx.clone()
            _score = score.clone()
            if _score.sum() < 1e-6:
                _score += 1
            _score[log_x_idx != self.num_classes - 1] = 0 # 标记不为mask的位置

            for i in range(log_x.shape[0]):
                n_sample = min(to_sample - sampled[i], max_sample_per_step) # 需要采样的
                if to_sample - sampled[i] - n_sample == 1:
                    n_sample = to_sample - sampled[i]
                if n_sample <= 0:
                    continue
                sel = torch.multinomial(_score[i], n_sample) # 根据权重采样，结果为index
                out2_idx[i][sel] = out_idx[i][sel]
                sampled[i] += ((out2_idx[i] != self.num_classes - 1).sum() - (log_x_idx[i] != self.num_classes - 1).sum()).item()

            out = index_to_log_onehot(out2_idx, self.num_classes)
        else:
            # Gumbel sample
            out = self.log_sample_categorical(model_log_prob)
            sampled = [self.prior_ps] * log_x.shape[0] # we change the 1024  to self.prior_ps

        if to_sample is not None:
            return out, sampled
        else:
            return out

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30) # 产生一定的噪声
        # print('logits ', logits)
        # print('gumbel_noise ', gumbel_noise)
        # assert 1==2
        sample = (gumbel_noise + logits).argmax(dim=1) # 每行最大值所在的index
        #print('sample ',sample.shape) # [2, 1024]
        log_sample = index_to_log_onehot(sample, self.num_classes) # 又把index转为log one-hot
        return log_sample

    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t) # 从x_0开始，往前走t步 (马尔科夫链),获得logq(xt|x0)
        # print('log_EV_qxt_x0 ',log_EV_qxt_x0.shape) # [2, 2888, 1024]
        # print('log_EV_qxt_x0 ',log_EV_qxt_x0)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0) # 根据概率分布，进行采样
        # print('log_sample ',log_sample)
        # assert 1==2
        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            # print('self.Lt_count ',self.Lt_count)
            if not (self.Lt_count > 10).all(): # 当矩阵里每个值都大于10时，才不使用 uniform 采样
                # print('use uniform... ')
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            #print('Lt_sqrt ',Lt_sqrt)
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True) # 采index 权重大的，采到的几率就越大
            # input张量可以看成一个权重张量，每一个元素代表其在该行中的权重。如果有元素为0，那么在其他不为0的元素被取干净之前，这个元素是不会被取到的。
            pt = pt_all.gather(dim=0, index=t) # 根据index,找到对应的值
            # print('pt ',pt)
            # assert 1==2
            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long() #从[0,num_timesteps]随机产生b个数
            # print('t ',t)
            pt = torch.ones_like(t).float() / self.num_timesteps # 概率一直都是0.01?
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond_emb, x_mask=None, cond_emb_mask=None, is_train=True):                       # get the KL loss
        b, device = x.size(0), x.device
        assert self.loss_type == 'vb_stochastic'
        x_start = x # (b, N)
        t, pt = self.sample_time(b, device, 'importance') # 时间采样
        #print('x_start ', x_start.shape)
        log_x_start = index_to_log_onehot(x_start, self.num_classes) # 将数值代表，转换为由one-hot向量组成的矩阵,其中每个向量最大值所在的索引就是原始的值
        #print('log_x_start ', log_x_start.shape)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t) # 通过采样获得 log_xt, 随机采得
        xt = log_onehot_to_index(log_xt) # get b, N
        #print('xt  ',xt)
        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, cond_emb, x_mask, cond_emb_mask, t=t)            # P_theta(x0|xt)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)
        #print('log_model_prob ', log_model_prob)
        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon) # 获得预测的x_0
        #print('x0_recon ', x0_recon)
        x0_real = x_start # 真实值

        xt_1_recon = log_onehot_to_index(log_model_prob) # 直接采样x_(t-1), 与 x(t)相同的数量
        xt_recon = log_onehot_to_index(log_xt) # 
        x_mask_repeat = x_mask.unsqueeze(1).repeat(1, self.n_q, 1) # (B, Len) --> (B, n_q, len)
        x_mask = x_mask_repeat.reshape(x_mask_repeat.shape[0], -1) # B, Len*n_q
        for index in range(t.size()[0]):
            this_t = t[index].item()
            tmp_mask = ~x_mask[index] # 获得当前样本的mask值
            tmp_x0_recon = x0_recon[index][tmp_mask] # 只保留非mask部分
            tmp_x0_real = x0_real[index][tmp_mask]
            same_rate = (tmp_x0_recon == tmp_x0_real).sum().cpu()/tmp_x0_real.shape[0] # 
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9 # 平均移动更新
            tmp_xt_1_recon = xt_1_recon[index][tmp_mask]
            tmp_xt_recon = xt_recon[index][tmp_mask]
            same_rate = (tmp_xt_1_recon == tmp_xt_recon).sum().cpu()/tmp_xt_recon.shape[0]
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9

        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t) # using true label to calculate
        #print('log_true_prob ', log_true_prob)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        #print('kl ', kl)
        mask_region = (xt == self.num_classes-1).float() # xt 中被mask的区域
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight*(~x_mask) # 去掉padding部分
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob) # 分类的概率 e_0
        #print('decoder_nll ', decoder_nll.shape)
        # assert 1==2
        decoder_nll = decoder_nll * (~x_mask) # mask padding部分
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float() # t为0, p(x_0|x_1,y)
        kl_loss = mask * decoder_nll + (1. - mask) * kl # t为0时，直接计算分类损失,否则计算kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach() # 记录下kl
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2)) # 记录加的次数,也可理解为选择了时间步t的次数

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt  # pt 代表得到采样时间的概率
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train==True:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            #print('kl_aux ', kl_aux)
            kl_aux = kl_aux * mask_weight*(~x_mask)
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = t/self.num_timesteps + 1.0
            else:
                addition_loss_weight = 1.0
            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            #print('loss2 ', loss2)
            vb_loss += loss2
        #assert 1==2
        return log_model_prob, vb_loss

    @property
    def device(self):
        return self.transformer.content_emb.pos_emb.weight.device

    def parameters(self, recurse=True, name=None):
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
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}# if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )
            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def forward(
            self, 
            input, 
            return_loss=True, 
            return_logits=True,
            is_train=True,
            **kwargs):
        #{'condition_text_embed': cond, 'condition_mel_mask': cond_mask, 'content_token': }
        # the input is prompt_semantic, target_semantic, prompt_acoustic, target_acoustic
        if kwargs.get('autocast') == True:
            self.amp = True
        batch_size = input['target_acoustics'].shape[0]
        device = input['target_acoustics'].device
        # 1) get embeddding for condition and content     prepare input
        content_token = input['target_acoustics'].reshape(batch_size, -1)
        content_token_mask = None # 目前先不使用 mask,因为我们直接padding eos
        content_token_mask = input['x_mask']
        # print('sample_image ',sample_image.shape)
        # cont_emb = self.content_emb(sample_image)
        cond_emb = {}
        cond_emb_mask = None # condition mask
        cond_emb['prompt_semantics'] = input['prompt_semantics']
        cond_emb['prompt_acoustics'] = input['prompt_acoustics']
        cond_emb['target_semantics'] = input['target_semantics']
        # now we get cond_emb and sample_image
        if self.learnable_cf: # 目前先不用
            # b = torch.rand(cond_emb.shape[0]) > 0.1 # generate randomly 
            # is_empty_text = torch.logical_not(b.to(device)).unsqueeze(1).unsqueeze(2).repeat(1, cond_emb.shape[1], 256)
            # #print('is_empty_text ', is_empty_text)
            # # torch.where(condition，a，b)其中输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出。
            # cond_emb = torch.where(is_empty_text, self.empty_text_embed[:cond_emb.shape[1],:].unsqueeze(0).repeat(cond_emb.shape[0], 1, 1), cond_emb.type_as(self.empty_text_embed))
            # cond_emb = cond_emb.float()
            pass
        if is_train == True:
            log_model_prob, loss = self._train_loss(content_token, cond_emb, content_token_mask, cond_emb_mask)
            loss = loss.sum()/(content_token.size()[0] * content_token.size()[1]) # ? mask
        # 4) get output, especially loss
        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)
        if return_loss:
            out['loss'] = loss 
        self.amp = False
        return out

    def sample(
            self,
            batch,
            filter_ratio = 0.5,
            temperature = 1.0,
            return_att_weight = False,
            return_logits = False,
            content_logits = None,
            print_log = True,
            **kwargs):
        real_content = batch['target_acoustics']
        batch_size = real_content.shape[0]
        device = self.log_at.device
        start_step = int(self.num_timesteps * filter_ratio) # 100*filter_ratio
        condition = {}
        condition['prompt_semantics'] = batch['prompt_semantics']
        condition['prompt_acoustics'] = batch['prompt_acoustics']
        condition['target_semantics'] = batch['target_semantics']
        content_token_mask = batch['x_mask']
        # print('content_token_mask ', content_token_mask.shape)
        # print(batch['target_semantics'].shape)
        # print(batch['target_acoustics'].shape)
        condition_mask = None
        # get cont_emb and cond_emb
        if start_step == 0: # when filter_ratio==0
            # use full mask sample
            # Note that this part only support mask, mask and uniform strategies, if you use uniform strategy
            predict_len = batch['target_acoustics'].shape[-1]*self.n_q
            self.prior_ps = predict_len # unpdate prior_ps and n_sample
            self.update_n_sample(predict_len) #  
            zero_logits = torch.zeros((batch_size, self.num_classes-1, predict_len),device=device) #b,256,265
            one_logits = torch.ones((batch_size, 1, predict_len),device=device) #b,1,265
            mask_logits = torch.cat((zero_logits, one_logits), dim=1) # 每个token全是mask
            log_z = torch.log(mask_logits)
            start_step = self.num_timesteps
            with torch.no_grad():
                for diffusion_index in range(start_step-1, -1, -1): # 99,0
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long) #用diffusion_index填充，shape=(b,)
                    sampled = [0] * log_z.shape[0] # 初始时 sampled 全设为0
                    #print('diffusion_index ', diffusion_index, len(self.n_sample))
                    while min(sampled) < self.n_sample[diffusion_index]:
                        log_z, sampled = self.p_sample(log_z, condition, t, content_token_mask, 
                                            condition_mask, sampled, self.n_sample[diffusion_index])     # log_z is log_onehot
        else:
            print('erroe, we must sample from zero')
            assert 1==2
        content_token = log_onehot_to_index(log_z) # transfer from one-hot to index
        output = {'pre_content_token': content_token} # return the predict content_token
        if return_logits: # false
            output['logits'] = torch.exp(log_z)
        return output
