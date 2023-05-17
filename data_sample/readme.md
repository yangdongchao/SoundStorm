### We provide the data structure for readers can ready their own data
We use the pre-trained HuBERT model to extract semantic token, and set the 1000 cluster. Pre-trained models can be found from: https://github.com/Rongjiehuang/TranSpeech

For acoustic token, you can use torch.save to save a dict: {item_name: acoustic_tokens}