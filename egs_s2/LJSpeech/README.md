1. Firstly, you should download `mhubert_base_vp_en_es_fr_it3.pt` and` mhubert_base_vp_en_es_fr_it3_L11_km1000.bin` from [Rongjiehuang/TranSpeech](https://github.com/Rongjiehuang/TranSpeech) to `pretrained_model/mhubert`

Then we can use Semantic_tokenizer to extract semantic token

### Preprocess data
```bash
./run.sh --stage 0 --stop-stage 0
```
### Train
```bash
./run.sh --stage 1 --stop-stage 1
```
