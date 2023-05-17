## SoundStorm: Efficient Parallel Audio Generation (wip)
Implementation of <a href="https://google-research.github.io/seanet/soundstorm/examples/">SoundStorm</a>, a Parallel Audio Generation out of Google Research, in Pytorch <br>

We first provide the first version code by ourselves. We directly use a mask-based discrete diffusion to implement this, which enjoys the same process as Google's paper. The details, please refer to our paper, InsturctTTS: https://arxiv.org/pdf/2301.13662.pdf <br>

We will update the second version based MASKGIT, which keep the same as SoundStorm.

## Overview
Following the paper, we use HuBERT to extract semantic tokens, and then using semantic token as condition to predict all of the acoustic tokens in parallel. Different with SoundStrom to use sum average the multiple codebook, we use shallow u-net to combine different codebook. For AudioCodec, we use the open source AcademiCodec https://github.com/yangdongchao/AcademiCodec

## Prepare dataset
Please refer to data_sample folder to understood how to prepare the dataset.

## Training
- Firtsly, prepare your data
- bash start/start.sh

## Inference
- Firstly, revise evaluation/generate_samples_batch.py based on your model.
- python generate_samples_batch.py

## Reference

```bibtex
@article{yang2023instructtts,
  title={InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt},
  author={Yang, Dongchao and Liu, Songxiang and Huang, Rongjie and Lei, Guangzhi and Weng, Chao and Meng, Helen and Yu, Dong},
  journal={arXiv preprint arXiv:2301.13662},
  year={2023}
}
```

```bibtex
@article{google_soundstorm,
  title={SoundStorm: Efficient Parallel Audio Generation},
  author={Zal´an Borsos, Matt Sharifi, Damien Vincent, Eugene Kharitonov, Neil Zeghidour, Marco Tagliasacchi},
  journal={arXiv preprint arXiv:2305},
  year={2023}
}
```

```bibtex
@article{yang2023hifi,
  title={HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec},
  author={Yang, Dongchao and Liu, Songxiang and Huang, Rongjie and Tian, Jinchuan and Weng, Chao and Zou, Yuexian},
  journal={arXiv preprint arXiv:2305.02765},
  year={2023}
}
```