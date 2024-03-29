# scene_detection_pytorch

## My research repository  

Scene Detection using Multimodal Deep Triplet Networks

alpha=36のほうが良いっぽい

# TODO List
+ [ ]: eval metricsの計算

+ [ ]: 各シーンのフレーム一覧

+ [x]: place-365での特徴抽出
+ [x]: 10sec Audioでの特徴抽出
+ [x]: TripletNetDatasetの実装
+ [x]: trainデータをmergeするコード(dataloader内)
+ [x]: textの前処理
+ [x]: tensorboard Embedding
+ [x]: TripletNetのT-SNE Embedding
+ [x]: DataLoaderの実装
+ [x]: SiameseNetの実装
+ [x]: 学習フェーズの実装
+ [x]: Compact Bilinear Poolingの実装

# Memo...
- LossのL2項
- distanceのl2 distance
- Online TripletNetworks

# Code Structure
- **data_preprocessing.py**
  - データの前処理を行う
- **eda.py**
  - データのEDAを行う

## Reference 
・PyTorch transforms/Dataset/DataLoaderの基本動作を確認する:
https://qiita.com/takurooo/items/e4c91c5d78059f92e76d

・pytorch siamese-triplet:
https://github.com/adambielski/siamese-triplet

・PyTorch/TorchVisionで複数の入力をモデルに渡したいケース:
https://blog.shikoan.com/torchvision-multiple-inputs/

・pytorch siamese:
https://vaaaaaanquish.hatenablog.com/entry/2019/02/23/214036

・BERT Tutorial:
http://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#3-extracting-embeddings

・VGGish:
https://github.com/harritaylor/torchvggish

・Places-365 pre-trained models:
https://github.com/CSAILVision/places365

https://zhuanlan.zhihu.com/p/85438252


## Usage TensorBoard
```
ssh username@IP_Address -L 6006:localhost:6006
tensorboard --logdir <log_dir>

```
