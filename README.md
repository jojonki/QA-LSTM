# QA-LSTM

LSTM-based Deep Learning Models for Non-factoid Answer Selection
Ming Tan, Cicero dos Santos, Bing Xiang, Bowen Zhou, ICLR 2016
https://arxiv.org/abs/1511.04108

This repo contains the implementation of QA-LSTM in PyTorch. Currently only concludes a basic QA-LSTM model.


## Requirements
- pytorch 0.3.0
- tqdm
- [insuranceQA dataset V2](https://github.com/shuzi/insuranceQA/tree/master/V2)

## TODOs
- [ ] Dropout before cosine similality matching
- [ ] 100-dim wordembedding and fine-tune
- [ ] Add CNN and attention models
- [ ] Performance analysis
- [ ] maxpooling

