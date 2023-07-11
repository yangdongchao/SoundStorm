### How to use?
Firstly, you should download hubert from https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

Then we can use Semantic_tokenizer to extract semantic token

About how to learn a new kmeans, please check https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans

We can use `dump_hubert_feature.py` and `learn_kmeans.py` in `exps` file to learn a new `*.bin` model with different `n_clusters`