import argparse
import os
import time

import torch
import torch.distributed as dist
from dataloader import build_train_and_valid_data_iterator
from dataloader import get_collect_cn_function
from distributed.launch import launch
from speartts_model import CoarseTransformer
from speartts_model import CoarseTransformerWrapper
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from utils import Logger
from utils import seed_everything
#sys.path.append('/apdcephfs/share_1316500/donchaoyang/audio_framework/SoundStream')
NODE_RANK = os.environ['INDEX'] if 'INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = (os.environ['CHIEF_IP'],
                            22275) if 'CHIEF_IP' in os.environ else (
                                "127.0.0.1", 29500)
#MASTER_ADDR, MASTER_PORT = ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
NUM_NODE = os.environ['HOST_NUM'] if 'HOST_NUM' in os.environ else 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_node',
        type=int,
        default=NUM_NODE,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--ngpus_per_node',
        type=int,
        default=8,
        help='number of gpu on one node')
    parser.add_argument(
        '--node_rank',
        type=int,
        default=NODE_RANK,
        help='node rank for distributed training')
    parser.add_argument(
        '--dist_url',
        type=str,
        default=DIST_URL,
        help='url used to set up distributed training')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU id to use. If given, only the specific gpu will be'
        ' used, and ddp will be disabled')
    parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    # args for random
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='seed for initializing training. ')
    parser.add_argument(
        '--cudnn_deterministic',
        action='store_true',
        help='set cudnn.deterministic True')
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='use tensorboard for logging')
    # args for training
    parser.add_argument(
        '--N_EPOCHS', type=int, default=100, help='Total training epoch')
    parser.add_argument(
        '--st_epoch', type=int, default=0, help='start training epoch')
    parser.add_argument(
        '--global_step', type=int, default=0, help='record the global step')
    parser.add_argument(
        '--num_semantic_token',
        type=int,
        default=1000,
        help='the number of semantic token')
    parser.add_argument(
        '--codebook_size',
        type=int,
        default=1024,
        help='the number of codebook_size')
    parser.add_argument(
        '--num_quantizer',
        type=int,
        default=3,
        help='the number of num_quantizer')
    parser.add_argument(
        '--transformer_dim',
        type=int,
        default=512,
        help='the number of transformer_dim')
    parser.add_argument(
        '--transformer_depth',
        type=int,
        default=8,
        help='the number of transformer_depth')

    parser.add_argument('--BATCH_SIZE', type=int, default=1, help='batch size')
    parser.add_argument(
        '--PATH',
        type=str,
        default='/apdcephfs/share_1316500/donchaoyang/audio_framework/SoundStream2/model_path/',
        help='batch size')
    parser.add_argument('--sr', type=int, default=16000, help='sample rate')
    parser.add_argument(
        '--print_freq', type=int, default=500, help='the print number')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/apdcephfs/share_1316500/donchaoyang/audio_framework/SoundStream2/log',
        help='log save path')
    parser.add_argument(
        '--train_data_path', type=str, default='', help='training data')
    parser.add_argument(
        '--valid_data_path', type=str, default='', help='training data')
    parser.add_argument(
        '--resume', action='store_true', help='whether re-train model')
    parser.add_argument(
        '--resume_path',
        type=str,
        default='/apdcephfs/share_1316500/donchaoyang/audio_framework/data/siri_renshe_1024utts',
        help='resume_path')
    args = parser.parse_args()
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if args.resume:
        args.PATH = args.resume_path  # direcly use the old model path
    else:
        args.PATH = os.path.join(args.PATH, time_str)
    args.save_dir = os.path.join(args.save_dir, time_str)
    os.makedirs(args.PATH, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()

# 设置 rank 对应的 split, 这里直接是就是 rank 号，我们可以设置一个字典
# get data_path_for_rank
def get_content(args):
    content = 'semantic:' + '/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/dongchao/code5/SpearTTS_32gpu_test/data_infor/semantic/gpu_' + str(
        args.global_rank) + '.pth'
    content = content + ",acoustic:" + '/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/dongchao/code5/SpearTTS_32gpu_test/data_infor/acoustic/gpu_' + str(
        args.global_rank) + '.pth'
    return content


def main():
    args = get_args()
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)
    if args.num_node == 1:
        args.dist_url == "auto"
    else:
        assert args.num_node > 1
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.num_node  #
    launch(
        main_worker,
        args.ngpus_per_node,
        args.num_node,
        args.node_rank,
        args.dist_url,
        args=(args, ))


def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1
    #CUDA_VISIBLE_DEVICES = int(args.local_rank)
    # build model 
    coarse_transformer = CoarseTransformer(
        num_semantic_tokens=args.num_semantic_token,
        codebook_size=args.codebook_size,
        num_coarse_quantizers=args.num_quantizer,
        dim=args.transformer_dim,
        layers=args.transformer_depth)
    audio_gpt = CoarseTransformerWrapper(transformer=coarse_transformer)
    logger = Logger(args)
    if args.distributed:
        audio_gpt = torch.nn.SyncBatchNorm.convert_sync_batchnorm(audio_gpt)
    # torch.distributed.barrier()
    args.device = torch.device('cuda', args.local_rank)
    audio_gpt.to(args.device)
    if args.distributed:
        audio_gpt = DDP(
            audio_gpt,
            device_ids=[args.local_rank],
            find_unused_parameters=True
        )  # device_ids=[args.local_rank], output_device=args.local_rank
    packed_data = get_content(args)
    #print('packed_data ', packed_data)
    collect_fn = get_collect_cn_function()
    train_loader, valid_loader = build_train_and_valid_data_iterator(
        args,
        packed_data,
        collect_fn,
        batch_scale=5000,
        batch_length_key='semantic')
    optimizer_g = torch.optim.AdamW(
        audio_gpt.parameters(), lr=3e-4, betas=(0.5, 0.9))
    lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g, gamma=0.999)
    if args.resume:
        latest_info = torch.load(args.resume_path + '/latest.pth')
        args.st_epoch = latest_info['epoch']
        audio_gpt.load_state_dict(latest_info['audio_gpt'])
        optimizer_g.load_state_dict(latest_info['optimizer_g'])
        lr_scheduler_g.load_state_dict(latest_info['lr_scheduler_g'])
    train(args, audio_gpt, train_loader, valid_loader, optimizer_g,
          lr_scheduler_g, logger)


def train(args, audio_gpt, train_loader, valid_loader, optimizer_g,
          lr_scheduler_g, logger):
    best_val_loss = float("inf")
    best_val_epoch = -1
    global_step = 0
    for epoch in range(args.st_epoch, args.N_EPOCHS + 1):
        audio_gpt.train()
        train_semantic_loss = 0.0
        train_acoustic_loss = 0.0
        k_iter = 0
        train_acc = 0.0
        # if args.distributed:
        #     train_loader.sampler.set_epoch(epoch)
        for (prompt_semantic, target_semantic, prompt_acoustic,
             target_acoustic) in tqdm(train_loader):
            # x = x.to(args.device)
            # print('prompt_semantic ', prompt_semantic.shape)
            # print('target_semantic ', target_semantic.shape)
            # print('prompt_acoustic ', prompt_acoustic.shape)
            # print('target_acoustic ', target_acoustic.shape)
            prompt_semantic = prompt_semantic.to(args.device)
            target_semantic = target_semantic.to(args.device)
            prompt_acoustic = prompt_acoustic.to(args.device)
            target_acoustic = target_acoustic.to(args.device)
            total_loss, target_acoustic_loss, target_semantic_loss, acc = audio_gpt(
                prompt_semantic,
                target_semantic,
                prompt_acoustic,
                target_acoustic,
                return_loss=True)
            train_semantic_loss += target_semantic_loss.item()
            train_acoustic_loss += target_acoustic_loss.item()
            train_acc += acc.item()
            optimizer_g.zero_grad()
            total_loss.backward()
            optimizer_g.step()
            k_iter += 1
            message = '<epoch:{:d}, iter:{:d}, total_loss:{:.4f}, target_acoustic_loss:{:.4f}, target_semantic_loss:{:.4f}, acc:{:.4f}'.format(
                epoch, k_iter,
                total_loss.item(),
                target_acoustic_loss.item(),
                target_semantic_loss.item(), acc.item())
            if k_iter % args.print_freq == 0:
                logger.log_info(message)
        lr_scheduler_g.step()
        message = '<epoch:{:d}, <train_semantic_loss:{:.4f}, train_acoustic_loss:{:.4f}, train_acc:{:.4f}>'.format(
            epoch, train_semantic_loss / len(train_loader), train_acoustic_loss
            / len(train_loader), train_acc / len(train_loader))
        logger.log_info(message)
        with torch.no_grad():
            audio_gpt.eval()
            val_semantic_loss = 0.0
            val_acoustic_loss = 0.0
            val_acc = 0.0
            for (prompt_semantic, target_semantic, prompt_acoustic,
                 target_acoustic) in tqdm(valid_loader):
                prompt_semantic = prompt_semantic.to(args.device)
                target_semantic = target_semantic.to(args.device)
                prompt_acoustic = prompt_acoustic.to(args.device)
                target_acoustic = target_acoustic.to(args.device)
                total_loss, target_acoustic_loss, target_semantic_loss, acc = audio_gpt(
                    prompt_semantic,
                    target_semantic,
                    prompt_acoustic,
                    target_acoustic,
                    return_loss=True)
                val_semantic_loss += target_semantic_loss.item()
                val_acoustic_loss += target_acoustic_loss.item()
                val_acc += acc.item()

            if dist.get_rank() == 0:
                best_model = audio_gpt.state_dict().copy()
                latest_model_audio_gpt = audio_gpt.state_dict().copy()
                if val_acoustic_loss < best_val_loss:
                    best_val_loss = val_acoustic_loss
                    best_val_epoch = epoch
                torch.save(best_model,
                           args.PATH + '/best_' + str(epoch) + '.pth')
                latest_save = {}
                latest_save['audio_gpt'] = latest_model_audio_gpt
                latest_save['epoch'] = epoch
                latest_save['optimizer_g'] = optimizer_g.state_dict()
                latest_save['lr_scheduler_g'] = lr_scheduler_g.state_dict()
                torch.save(latest_save, args.PATH + '/latest.pth')
            message = '<epoch:{:d}, val_semantic_loss:{:.4f}, val_acoustic_loss:{:.4f}, val_acc:{:.4f}, best_val_epoch:{:.4f}'.format(
                epoch, val_semantic_loss / len(valid_loader),
                val_acoustic_loss / len(valid_loader),
                val_acc / len(valid_loader), best_val_epoch)
            logger.log_info(message)


if __name__ == '__main__':
    main()
