# ------------------------------------------
# Diffsound
# written by Dongchao Yang
# based https://github.com/cientgu/VQ-Diffusion
# ------------------------------------------
import math
import os
import time

import numpy as np
import soundfile as sf
import torch
from soundstorm.s2.distributed.distributed import is_primary
from soundstorm.s2.distributed.distributed import reduce_dict
from soundstorm.s2.engine.ema import EMA
from soundstorm.s2.engine.lr_scheduler import ReduceLROnPlateauWithWarmup
from soundstorm.s2.utils.misc import format_seconds
from soundstorm.s2.utils.misc import get_model_parameters_info
from soundstorm.s2.utils.misc import instantiate_from_config
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
    AMP = True
except Exception:
    print('Warning: import torch.amp failed, so no amp will be used!')
    AMP = False

STEP_WITH_LOSS_SCHEDULERS = (ReduceLROnPlateauWithWarmup, ReduceLROnPlateau)


class Solver(object):
    def __init__(self, config, args, model, dataloader, logger, hificodec):
        self.config = config
        self.args = args
        self.model = model
        self.hificodec = hificodec

        self.dataloader = dataloader
        self.logger = logger

        self.max_epochs = config['solver']['max_epochs']
        self.save_epochs = config['solver']['save_epochs']
        self.save_iterations = config['solver'].get('save_iterations', -1)
        # 多少个 epoch 验证一次
        self.dev_epochs = config['solver'].get('dev_epochs', 2)
        # sample() 很耗时，需要 70s 
        # self.sample_epochs = self.dev_epochs * 2
        self.sample_epochs = 1

        assert isinstance(self.save_epochs, (int, list))
        assert isinstance(self.dev_epochs, (int, list))
        self.debug = config['solver'].get('debug', False)

        self.last_epoch = -1
        self.last_iter = -1
        self.epoch_time = 0

        self.total_iters = self.max_epochs * self.dataloader['train_iterations']

        self.ckpt_dir = os.path.join(args.output, 'checkpoint')
        self.audio_dir = os.path.join(args.output, 'audios')

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        # get grad_clipper
        if 'clip_grad_norm' in config['solver']:
            self.clip_grad_norm = instantiate_from_config(
                config['solver']['clip_grad_norm'])
        else:
            self.clip_grad_norm = None

        # get lr
        # none
        adjust_lr = config['solver'].get('adjust_lr', 'sqrt')
        # 3.0e-6
        base_lr = config['solver'].get('base_lr', 1.0e-4)
        # 若不存在则用 1e-4
        if adjust_lr == 'none':
            self.lr = base_lr
        # 平方调整
        elif adjust_lr == 'sqrt':
            self.lr = base_lr * math.sqrt(args.world_size *
                                          config['dataloader']['batch_size'])
        elif adjust_lr == 'linear':
            self.lr = base_lr * args.world_size * config['dataloader'][
                'batch_size']
        else:
            raise NotImplementedError(
                'Unknown type of adjust lr {}!'.format(adjust_lr))
        self.logger.log_info('Get lr {} from base lr {} with {}'.format(
            self.lr, base_lr, adjust_lr))

        if hasattr(model, 'get_optimizer_and_scheduler') and callable(
                getattr(model, 'get_optimizer_and_scheduler')):
            optimizer_and_scheduler = model.get_optimizer_and_scheduler(
                config['solver']['optimizers_and_schedulers'])
        else:
            optimizer_and_scheduler = self._get_optimizer_and_scheduler(
                config['solver']['optimizers_and_schedulers'])

        assert type(optimizer_and_scheduler) == type(
            {}), 'optimizer and schduler should be a dict!'
        self.optimizer_and_scheduler = optimizer_and_scheduler

        # configre for ema
        if 'ema' in config['solver'] and args.local_rank == 0:
            ema_args = config['solver']['ema']
            ema_args['model'] = self.model
            self.ema = EMA(**ema_args)
        else:
            self.ema = None
        self.logger.log_info(str(get_model_parameters_info(self.model)))
        #self.model.cuda() 
        self.model.to(self.args.local_rank)
        if args.local_rank == 0:
            self.hificodec.to(self.args.local_rank)
        self.device = self.model.device

        if self.args.distributed:
            # the next line change arg.gup to args.local_rank
            self.logger.log_info('Distributed, begin DDP the model...')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                find_unused_parameters=True)
            self.logger.log_info('Distributed, DDP model done!')
        # prepare for amp
        self.args.amp = self.args.amp and AMP
        if self.args.amp:
            self.scaler = GradScaler()
            self.logger.log_info('Using AMP for training!')

        self.logger.log_info(
            "global rank {}: prepare solver done!".format(
                self.args.global_rank),
            check_primary=False)

    def _get_optimizer_and_scheduler(self, op_sc_list):
        optimizer_and_scheduler = {}
        for op_sc_cfg in op_sc_list:
            op_sc = {
                'name': op_sc_cfg.get('name', 'none'),
                'start_epoch': op_sc_cfg.get('start_epoch', 0),
                'end_epoch': op_sc_cfg.get('end_epoch', -1),
                'start_iteration': op_sc_cfg.get('start_iteration', 0),
                'end_iteration': op_sc_cfg.get('end_iteration', -1),
            }
            if op_sc['name'] == 'none':
                # parameters = self.model.parameters()
                parameters = filter(lambda p: p.requires_grad,
                                    self.model.parameters())
            else:
                # NOTE: get the parameters with the given name, the parameters() should be overide
                parameters = self.model.parameters(name=op_sc['name'])

            # build optimizer
            op_cfg = op_sc_cfg.get('optimizer',
                                   {'target': 'torch.optim.SGD',
                                    'params': {}})
            if 'params' not in op_cfg:
                op_cfg['params'] = {}
            if 'lr' not in op_cfg['params']:
                op_cfg['params']['lr'] = self.lr
            op_cfg['params']['params'] = parameters
            optimizer = instantiate_from_config(op_cfg)
            op_sc['optimizer'] = {
                'module': optimizer,
                'step_iteration': op_cfg.get('step_iteration', 1)
            }
            assert isinstance(
                op_sc['optimizer']['step_iteration'],
                int), 'optimizer steps should be a integer number of iterations'

            # build scheduler
            if 'scheduler' in op_sc_cfg:
                sc_cfg = op_sc_cfg['scheduler']
                sc_cfg['params']['optimizer'] = optimizer
                # for cosine annealing lr, compute T_max
                if sc_cfg['target'].split('.')[-1] in [
                        'CosineAnnealingLRWithWarmup', 'CosineAnnealingLR'
                ]:
                    T_max = self.max_epochs * self.dataloader[
                        'train_iterations']
                    sc_cfg['params']['T_max'] = T_max
                scheduler = instantiate_from_config(sc_cfg)
                op_sc['scheduler'] = {
                    'module': scheduler,
                    'step_iteration': sc_cfg.get('step_iteration', 1)
                }
                if op_sc['scheduler']['step_iteration'] == 'epoch':
                    op_sc['scheduler']['step_iteration'] = self.dataloader[
                        'train_iterations']
            optimizer_and_scheduler[op_sc['name']] = op_sc
        return optimizer_and_scheduler

    def _get_lr(self, return_type='str'):

        lrs = {}
        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            lr = op_sc['optimizer']['module'].state_dict()['param_groups'][0][
                'lr']
            lrs[op_sc_n + '_lr'] = round(lr, 10)
        if return_type == 'str':
            lrs = str(lrs)
            lrs = lrs.replace('none', 'lr').replace('{', '').replace(
                '}', '').replace('\'', '')
        elif return_type == 'dict':
            pass
        else:
            raise ValueError('Unknow of return type: {}'.format(return_type))
        return lrs

    def hificodec_decode(self, acoustic_token, rescale=True):
        # acoustic_token 应该只有 1025 没有 1024
        # acoustic_token 末尾的补零 (1025) 部分会生成高频噪声
        acoustic_token = np.clip(acoustic_token, 0, 1023)
        acoustic_token = torch.tensor(acoustic_token).cuda(self.args.local_rank)
        acoustic_token = acoustic_token.transpose(0, 1).unsqueeze(0)
        # VQVAE.forward()
        wav = self.hificodec(acoustic_token)
        wav = wav.detach().squeeze().cpu().numpy()
        limit = 0.99
        if rescale:
            mx = np.abs(wav).max()
            wav = wav * min(limit / mx, 1)
        else:
            wav = wav.clip(-limit, limit)
        return wav

    def sample(self,
               batch,
               phase='train',
               step_type='iteration',
               gen_audio=True):
        tic = time.time()
        self.logger.log_info('Begin to sample...')
        step = self.last_iter if step_type == 'iteration' else self.last_epoch
        sample_rate = 16000
        save_path = self.audio_dir
        save_name_gt = save_path + '/gt.npy'
        # shape (100, 4, 354)
        content_gt = batch['target_acoustics']
        codes_np_gt = content_gt.detach().cpu().numpy()
        # 重复写入，如果判断是否存在的话可能会因为多卡导致 lood 一个不完整的 npy
        np.save(save_name_gt, codes_np_gt)
        if gen_audio:
            # only save the wav of first 5 in batch
            # target wav in batch is reverse sorted with length
            audio_indexs = []
            for i in range(min(10, content_gt.size(0))):
                save_name_gt_wav = save_path + '/gt_' + str(i) + '.wav'
                acoustic_token_gt = codes_np_gt[i]
                # all Nq have same pad length
                acoustic_token_gt_0_list = acoustic_token_gt[0].tolist()
                index = acoustic_token_gt_0_list.index(
                    1025) if 1025 in acoustic_token_gt_0_list else -1
                audio_indexs.append(index)
                # clip wav via pad value (1025)
                acoustic_token_gt = acoustic_token_gt[:, :index]
                wav_gt = self.hificodec_decode(acoustic_token_gt)
                sf.write(save_name_gt_wav, wav_gt, sample_rate)
                self.logger.add_audio(
                    tag='gt/audio/' + str(i),
                    snd_tensor=wav_gt,
                    global_step=step,
                    sample_rate=sample_rate)

        if self.ema:
            self.ema.modify_to_inference()
            suffix = '_ema'
        else:
            suffix = ''
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        with torch.no_grad():
            if self.debug is False:
                if self.args.amp:
                    with autocast():
                        samples = model.infer_one(batch=batch)
                else:
                    samples = model.infer_one(batch=batch)
            else:
                samples = model.infer_one(batch=batch[0].cuda())

            # shape (100, 1416)
            content = samples['token_pred']
            # hificodec decode 需要 [B, T, Nq]
            # (100, 4, 354), [B, 4, T]
            codes = content.reshape(content.shape[0], 4, -1)
            codes_np = codes.detach().cpu().numpy()
            name_prefix = '/epoch_' + str(
                self.last_epoch) + '_last_iter_' + str(self.last_iter)
            save_name = save_path + name_prefix + '.npy'

            # self.hificodec
            np.save(save_name, codes_np)
            if gen_audio:
                for i in range(min(10, content_gt.size(0))):
                    save_name_wav = save_path + name_prefix + '_' + str(
                        i) + '.wav'
                    acoustic_token = codes_np[i]
                    index = audio_indexs[i]
                    # clip wav via pad value (1025) get from acoustic_token_gt
                    acoustic_token = acoustic_token[:, :index]
                    wav = self.hificodec_decode(acoustic_token)
                    sf.write(save_name_wav, wav, sample_rate)
                    self.logger.add_audio(
                        tag='gen/audio/' + str(i),
                        snd_tensor=wav,
                        global_step=step,
                        sample_rate=sample_rate)

        if self.ema:
            self.ema.modify_to_train()
        # 74s 为什么这么耗时
        self.logger.log_info(
            'Sample done, time: {:.2f} s'.format(time.time() - tic))

    def step(self, batch, phase='train'):
        loss = {}
        if self.debug is False:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()
        else:
            batch = batch[0].cuda()
        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            if phase == 'train':
                # this part is nothing
                # check if this optimizer and scheduler is valid in this iteration and epoch
                # pass
                if op_sc['start_iteration'] > self.last_iter:
                    continue
                # pass
                if op_sc['end_iteration'] > 0 and op_sc[
                        'end_iteration'] <= self.last_iter:
                    continue
                # pass
                if op_sc['start_epoch'] > self.last_epoch:
                    continue
                # pass 
                if op_sc['end_epoch'] > 0 and op_sc[
                        'end_epoch'] <= self.last_epoch:
                    continue

            input = {
                'batch': batch,
                'return_loss': True,
                'step': self.last_iter,
            }
            if op_sc_n != 'none':
                input['name'] = op_sc_n

            if phase == 'train':
                if self.args.amp:
                    with autocast():
                        output = self.model(**input)
                else:
                    output = self.model(**input)
            else:
                with torch.no_grad():
                    if self.args.amp:
                        with autocast():
                            output = self.model(**input)
                    else:
                        output = self.model(**input)

            if phase == 'train':
                if op_sc['optimizer']['step_iteration'] > 0 and (
                        self.last_iter + 1
                ) % op_sc['optimizer']['step_iteration'] == 0:
                    # to zero
                    op_sc['optimizer']['module'].zero_grad()
                    # nothing, we donot use amp now
                    if self.args.amp:
                        self.scaler.scale(output['loss']).backward()
                        if self.clip_grad_norm:
                            self.clip_grad_norm(self.model.parameters())
                        self.scaler.step(op_sc['optimizer']['module'])
                        self.scaler.update()
                    else:
                        # backward
                        output['loss'].backward()
                        if self.clip_grad_norm:
                            self.clip_grad_norm(self.model.parameters())
                        # update
                        op_sc['optimizer']['module'].step()
                # 更新 lr
                if 'scheduler' in op_sc:
                    if op_sc['scheduler']['step_iteration'] > 0 and (
                            self.last_iter + 1
                    ) % op_sc['scheduler']['step_iteration'] == 0:
                        # 每个 iter 调用一次
                        if isinstance(op_sc['scheduler']['module'],
                                      STEP_WITH_LOSS_SCHEDULERS):
                            op_sc['scheduler']['module'].step(
                                output.get('loss'))
                        else:
                            op_sc['scheduler']['module'].step()
                # update ema model
                if self.ema:
                    self.ema.update(iteration=self.last_iter)

            loss[op_sc_n] = {
                k: v
                for k, v in output.items() if ('loss' in k or 'acc' in k)
            }
        return loss

    def save(self, force=False):
        if is_primary():
            # save with the epoch specified name
            if self.save_iterations > 0:
                if (self.last_iter + 1) % self.save_iterations == 0:
                    save = True
                else:
                    save = False
            else:
                if isinstance(self.save_epochs, int):
                    save = (self.last_epoch + 1) % self.save_epochs == 0
                else:
                    save = (self.last_epoch + 1) in self.save_epochs

            if save or force:
                state_dict = {
                    'last_epoch':
                    self.last_epoch,
                    'last_iter':
                    self.last_iter,
                    'model':
                    self.model.module.state_dict()
                    if isinstance(self.model,
                                  torch.nn.parallel.DistributedDataParallel)
                    else self.model.state_dict()
                }
                if self.ema:
                    state_dict['ema'] = self.ema.state_dict()
                if self.clip_grad_norm:
                    state_dict[
                        'clip_grad_norm'] = self.clip_grad_norm.state_dict()

                # add optimizers and schedulers
                optimizer_and_scheduler = {}
                for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
                    state_ = {}
                    for k in op_sc:
                        if k in ['optimizer', 'scheduler']:
                            op_or_sc = {
                                kk: vv
                                for kk, vv in op_sc[k].items() if kk != 'module'
                            }
                            op_or_sc['module'] = op_sc[k]['module'].state_dict()
                            state_[k] = op_or_sc
                        else:
                            state_[k] = op_sc[k]
                    optimizer_and_scheduler[op_sc_n] = state_

                state_dict['optimizer_and_scheduler'] = optimizer_and_scheduler
                # save per save_epochs
                if save:
                    save_path = os.path.join(
                        self.ckpt_dir, '{}e_{}iter.pth'.format(
                            str(self.last_epoch).zfill(6), self.last_iter))
                    torch.save(state_dict, save_path)
                    self.logger.log_info('saved in {}'.format(save_path))

                # save with the last name
                # save per epoch
                save_path = os.path.join(self.ckpt_dir, 'last.pth')
                torch.save(state_dict, save_path)
                self.logger.log_info('saved in {}'.format(save_path))

    def resume(
            self,
            path=None,  # The path of last.pth
            load_optimizer_and_scheduler=True,  # whether to load optimizers and scheduler
            load_others=True  # load other informations
    ):
        if path is None:
            path = os.path.join(self.ckpt_dir, 'last.pth')
        if os.path.exists(path):
            state_dict = torch.load(
                path, map_location='cuda:{}'.format(self.args.local_rank))
            if load_others:
                self.last_epoch = state_dict['last_epoch']
                self.last_iter = state_dict['last_iter']

            if isinstance(self.model,
                          torch.nn.parallel.DistributedDataParallel):
                try:
                    self.model.module.load_state_dict(state_dict['model'])
                except Exception:
                    model_dict = self.model.module.state_dict()
                    temp_state_dict = {
                        k: v
                        for k, v in state_dict['model'].items()
                        if k in model_dict.keys()
                    }
                    model_dict.update(temp_state_dict)
                    self.model.module.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(state_dict['model'])

            if 'ema' in state_dict and self.ema:
                try:
                    self.ema.load_state_dict(state_dict['ema'])
                except Exception:
                    model_dict = self.ema.state_dict()
                    temp_state_dict = {
                        k: v
                        for k, v in state_dict['ema'].items()
                        if k in model_dict.keys()
                    }
                    model_dict.update(temp_state_dict)
                    self.ema.load_state_dict(model_dict)

            if 'clip_grad_norm' in state_dict and self.clip_grad_norm:
                self.clip_grad_norm.load_state_dict(
                    state_dict['clip_grad_norm'])
                self.clip_grad_norm.last_iter = self.last_iter
            # 这里恢复了 optimizer_and_scheduler 的一些参数，包含 ['optimizer_and_scheduler']['none']['scheduler']['module']['last_epoch']
            # handle optimizer and scheduler
            for op_sc_n, op_sc in state_dict['optimizer_and_scheduler'].items():
                for k in op_sc:
                    if k in ['optimizer', 'scheduler']:
                        for kk in op_sc[k]:
                            if kk == 'module' and load_optimizer_and_scheduler:
                                self.optimizer_and_scheduler[op_sc_n][k][
                                    kk].load_state_dict(op_sc[k][kk])
                            # such as step_iteration, ...
                            elif load_others:
                                self.optimizer_and_scheduler[op_sc_n][k][
                                    kk] = op_sc[k][kk]
                    # such as start_epoch, end_epoch, ....
                    elif load_others:
                        self.optimizer_and_scheduler[op_sc_n][k] = op_sc[k]
            print('succss', self.args.global_rank)
            self.logger.log_info('Resume from {}'.format(path))

    def train_epoch(self):
        self.model.train()
        self.last_epoch += 1

        epoch_start = time.time()
        itr_start = time.time()
        itr = -1
        for itr, batch in enumerate(self.dataloader['train_loader']):
            # (B, 1, T), B, T 动态
            # print("batch['prompt_semantics'].shape:",batch['prompt_semantics'].shape)
            data_time = time.time() - itr_start
            step_start = time.time()
            self.last_iter += 1
            loss = self.step(batch, phase='train')
            # logging info
            if self.logger and self.last_iter % self.args.log_frequency == 0:
                cur_iter_in_epoch = (self.last_iter - self.last_epoch *
                                     self.dataloader['train_iterations']
                                     ) % self.dataloader['train_iterations']
                info = 'Train: Epoch {}/{} iter {}/{}'.format(
                    self.last_epoch, self.max_epochs, cur_iter_in_epoch,
                    self.dataloader['train_iterations'])
                for loss_n, loss_dict in loss.items():
                    info += ' ||'
                    loss_dict = reduce_dict(loss_dict)
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    # info = info + ': Epoch {}/{} iter {}/{}'.format(self.last_epoch, self.max_epochs, self.last_iter%self.dataloader['train_iterations'], self.dataloader['train_iterations'])
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        self.logger.add_scalar(
                            tag='train/{}/{}'.format(loss_n, k),
                            scalar_value=float(loss_dict[k]),
                            global_step=self.last_iter)

                # log lr
                lrs = self._get_lr(return_type='dict')
                for k in lrs.keys():
                    lr = lrs[k]
                    self.logger.add_scalar(
                        tag='train/{}_lr'.format(k),
                        scalar_value=lrs[k],
                        global_step=self.last_iter)

                # add lr to info
                info += ' || {}'.format(self._get_lr())

                # add time consumption to info
                spend_time = time.time() - self.start_train_time
                forward_time = time.time() - step_start
                # 1 卡 -> n 卡，iter_time 不变
                iter_time = time.time() - itr_start
                epoch_time = time.time() - epoch_start
                info += ' || data_time: {dt}s | fbward_time: {fbt}s | iter_time: {it}s | epoch_time: {et}s| left_time: {lt}'.format(
                    dt=round(data_time, 1),
                    it=round(iter_time, 1),
                    fbt=round(forward_time, 1),
                    et=self.epoch_time,
                    # self.dataloader['train_iterations']: iter per epoch
                    # 1 卡 -> n 卡，self.max_epochs 不变，self.dataloader['train_iterations'] 为原来的 1/n，单卡显存占用不变
                    # max_token_one_batch 变为 n 倍，self.dataloader['train_iterations'] 为原来的 1/n，单卡显存占用变为 n 倍
                    lt=format_seconds(iter_time *
                                      (self.total_iters - self.last_iter)))
                self.logger.log_info(info)

            itr_start = time.time()
        self.epoch_time = time.time() - epoch_start

        # modify here to make sure dataloader['train_iterations'] is correct
        assert itr >= 0, "The data is too less to form one iteration!"
        self.dataloader['train_iterations'] = itr + 1

    def validate_epoch(self):
        if 'dev_loader' not in self.dataloader:
            val = False
        else:
            if isinstance(self.dev_epochs, int):
                # 能否整除
                val = (self.last_epoch + 1) % self.dev_epochs == 0
            else:
                val = (self.last_epoch + 1) in self.dev_epochs
        if val:
            self.model.eval()

            overall_loss = None
            first_batch = None
            # 求所有 dev batch loss 的均值
            for itr, batch in enumerate(self.dataloader['dev_loader']):
                if first_batch is None:
                    first_batch = batch
                # val 直接用 is_primary() 时卡到这里了
                loss = self.step(batch, phase='val')
                for loss_n, loss_dict in loss.items():
                    loss[loss_n] = reduce_dict(loss_dict)
                if overall_loss is None:
                    overall_loss = loss
                else:
                    for loss_n, loss_dict in loss.items():
                        for k, v in loss_dict.items():
                            overall_loss[loss_n][k] = (
                                overall_loss[loss_n][k] * itr + loss[loss_n][k]
                            ) / (itr + 1)

            if self.logger:
                info = ''
                for loss_n, loss_dict in overall_loss.items():
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    info += 'Eval: Epoch {}/{}'.format(self.last_epoch,
                                                       self.max_epochs)
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        self.logger.add_scalar(
                            tag='val/{}/{}'.format(loss_n, k),
                            scalar_value=float(loss_dict[k]),
                            global_step=self.last_iter)
                self.logger.log_info(info)
            if is_primary():
                # sample
                # 用的是第一个 batch 的数据
                if (self.last_epoch + 1) % self.sample_epochs == 0:
                    self.sample(first_batch, phase='val', step_type='iteration')

    def train(self):
        start_epoch = self.last_epoch + 1
        self.start_train_time = time.time()
        self.logger.log_info(
            'global rank {}: start training...'.format(self.args.global_rank),
            check_primary=False)

        for epoch in range(start_epoch, self.max_epochs):
            # self.validate_epoch()
            self.train_epoch()
            self.save(force=True)
            self.validate_epoch()
