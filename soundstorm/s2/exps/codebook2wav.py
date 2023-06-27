# convert acoustic_token (codebook) extracted by codec to wav by different num_quant
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from academicodec.models.encodec.net3 import SoundStream
from academicodec.models.encodec.test import remove_encodec_weight_norm
from academicodec.models.hificodec.vqvae import VQVAE
from soundstorm.utils import str2bool


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--codec_name",
        default="hificodec",
        type=str,
        help="name of codec, should in {hificodec, encodec} now")

    parser.add_argument(
        "--model_path", type=str, default='./HiFi-Codec-16k-320d')

    parser.add_argument(
        '--sr', type=int, default=16000, help='sample rate of model')

    # for HiFi-Codec
    parser.add_argument(
        "--config_path", type=str, default='./config_16k_320d.json')

    # for Encodec
    parser.add_argument(
        '--ratios',
        type=int,
        nargs='+',
        # probs(ratios) = hop_size, default for 16k_320d
        default=[8, 5, 4, 2],
        help='ratios of SoundStream / Encodec, shoud be set for different hop_size (32d, 320, 240d, ...)'
    )
    parser.add_argument(
        '--target_bandwidths',
        type=float,
        nargs='+',
        # default for 16k_320d
        default=[1, 1.5, 2, 4, 6, 12],
        help='target_bandwidths of net3.py')
    parser.add_argument(
        '--target_bw',
        type=float,
        # default for 16k_320d
        default=12,
        help='target_bw of net3.py')
    # default Nq of HiFi-Codec is 4 and cannot be reduced
    parser.add_argument(
        "--num_quant",
        type=int,
        # use all Nq by default
        default=100,
        help="number of quant when decoding encodec")

    parser.add_argument(
        "--input_path",
        type=str,
        default='path of codebook, should be *.pth (C, T)')
    parser.add_argument(
        "--rescale",
        type=str2bool,
        default=True,
        help="Automatically rescale the output to avoid clipping.")

    parser.add_argument("--output_dir", type=str, default='dir to save wav')

    args = parser.parse_args()

    if args.codec_name == 'hificodec':
        model = VQVAE(
            config_path=args.config_path,
            ckpt_path=args.model_path,
            with_encoder=True)
        model.cuda()
        model.generator.remove_weight_norm()
        model.encoder.remove_weight_norm()
        model.eval()

    elif args.codec_name == 'encodec':
        model = SoundStream(
            n_filters=32,
            D=512,
            ratios=args.ratios,
            sample_rate=args.sr,
            target_bandwidths=args.target_bandwidths)
        parameter_dict = torch.load(args.model_path)
        new_state_dict = {}
        # k 为 module.xxx.weight, v 为权重
        for k, v in parameter_dict.items():
            # 截取 `module.` 后面的 xxx.weight
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.cuda()
        remove_encodec_weight_norm(model)
        model.eval()

    else:
        print("Please input the right codec_name!")

    codec_extractor = model
    # # for batch['target_acoustics']
    # # 使用 gt.npy 可以正常合成
    # acoustic_token = np.load("/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/exp_ljspeech/default/audios/epoch_119_last_iter_2759.npy")
    # acoustic_token = acoustic_token[0]
    # # acoustic_token 应该只有 1025 没有 1024
    # # acoustic_token 末尾的补零 (1025) 部分会生成高频噪声
    # acoustic_token = np.clip(acoustic_token, 0, 1023)

    acoustic_token = np.load(args.input_path)
    acoustic_token = torch.tensor(acoustic_token).cuda()

    if args.codec_name == 'hificodec':
        # default Nq of HiFi-Codec is 4 and cannot be reduced
        args.num_quant = 4
        # (Nq, T) -> (1, T, Nq)
        acoustic_token = acoustic_token.transpose(0, 1).unsqueeze(0)
        # VQVAE.forward()
        wav = codec_extractor(acoustic_token)
        wav = wav.detach().squeeze().cpu().numpy()
    elif args.codec_name == 'encodec':
        # select coodebook by args.num_quant
        args.num_quant = min(args.num_quant, acoustic_token.shape[0])
        acoustic_token = acoustic_token[:args.num_quant, ...]
        # (Nq, T) -> (Nq, 1, T)
        acoustic_token = acoustic_token.unsqueeze(1)
        wav = codec_extractor.decode(acoustic_token)
        wav = wav.detach().squeeze().squeeze().cpu().numpy()
    else:
        print("Please input the right codec_name!")

    limit = 0.99
    if args.rescale:
        mx = np.abs(wav).max()
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clip(-limit, limit)
    utt_id = args.input_path.split("/")[-1].split(".")[0]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = utt_id + "_" + args.codec_name + "_" + str(
        args.num_quant) + "Nq.wav"
    output_path = output_dir / output_name
    sf.write(output_path, wav, args.sr)


if __name__ == "__main__":
    main()
