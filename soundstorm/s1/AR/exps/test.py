# test from dump file
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from soundstorm.s1.AR.data.dataset import Text2SemanticDataset
from soundstorm.s1.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from soundstorm.utils.io import load_yaml_config
from torch.utils.data import DataLoader


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="Run SoundStorm AR S1 model for test set.")

    parser.add_argument(
        '--config_file',
        type=str,
        default='conf/default.yaml',
        help='path of config file')

    # args for dataset
    parser.add_argument(
        '--test_semantic_path',
        type=str,
        default='dump/test/semantic_token.pth')
    parser.add_argument(
        '--test_phoneme_path', type=str, default='dump/test/phonemes.npy')

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='exp/default/ckpt/epoch=99-step=49000.ckpt',
        help='Checkpoint file of SoundStorm AR S1 model.')

    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = load_yaml_config(args.config_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 1
    hz = 50
    max_sec = config['data']['max_sec']

    # get dataset
    test_dataset = Text2SemanticDataset(
        phoneme_path=args.test_phoneme_path,
        semantic_path=args.test_semantic_path,
        # max_sec 需要与训练时保持一致，不然可能会效果不好，重复漏字等
        # 但是这里设置太短又会直接过滤掉太长的样本，为了防止被过滤掉，可以在 infer 的时候截断
        max_sec=100,
        max_sample=8,
        pad_val=config['data']['pad_val'])
    # get model
    t2s_model = Text2SemanticLightningModule.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, config=config)
    t2s_model.cuda()
    t2s_model.eval()

    # 获取 batch_size 条
    # 创建 DataLoader，并指定 collate_fn 函数
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate)

    item_names = test_dataset.__get_item_names__()

    # 逐批次读取数据, bs=1、shuffle=False 时可以用 __get_item_names__ 对应
    semantic_token_dict = {}
    for i, batch in enumerate(dataloader):
        # 要保证 bs = 1
        utt_id = item_names[i]
        if i == 0:
            print("utt_id:", utt_id)
            # bs > 1 时会补零
            # 与 validation_step() 保持一致
            semantic_len = batch['semantic_ids'].size(1)
            # 以 batch['semantic_ids'] 的前 150 个为 prompt
            # 多次合成，前 prompt_len 个是一样的，而且和 prompt 一样
            prompt_len = min(int(semantic_len * 0.5), 150)
            # 输入纯文本时 prompt 该输入什么？=> see t2s.py
            prompt = batch['semantic_ids'][:, :prompt_len]
            # # zero prompt => 也可以输出文本内容正确的 semantic token, 但是音色是乱的
            # 证明 semantic token 中还是包含了音色信息
            # prompt = torch.ones(
            #     batch['semantic_ids'].size(0), 1, dtype=torch.int32) * 0
            # print("prompt:", prompt)
            # print("prompt.shape:", prompt.shape)
            np.save(output_dir / 'prompt.npy', prompt.detach().cpu().numpy())

            st = time.time()
            with torch.no_grad():
                # calculate acc for test
                loss, acc = t2s_model.model.forward(
                    batch['phoneme_ids'].cuda(),
                    batch['phoneme_ids_len'].cuda(),
                    batch['semantic_ids'].cuda(),
                    batch['semantic_ids_len'].cuda())
                print("top_3_acc of this batch:", acc)
                pred_semantic = t2s_model.model.infer(
                    batch['phoneme_ids'].cuda(),
                    batch['phoneme_ids_len'].cuda(),
                    prompt.cuda(),
                    top_k=config['inference']['top_k'],
                    # hz * max_sec in train dataloader
                    # 生成的长度是 1002 应该是有一些 pad
                    early_stop_num=hz * max_sec)
                # bs = 1
                pred_semantic = pred_semantic[0]
            print(f'{time.time() - st} sec used in T2S')
            semantic_token = pred_semantic.detach().cpu().numpy()
            semantic_token_dict[utt_id] = semantic_token.astype(np.int16)
        else:
            break

    filename = output_dir / "semantic_token.npy"
    np.save(filename, semantic_token_dict)


if __name__ == "__main__":
    main()
