# scene_detection_pytorch
My research theme  

# TODO List
+ [ ]: textの前処理
+ [x]: AudioCNNの実装
+ [ ]: TripletNetの実装
+ [ ]: tensorboard Embedding
+ [x]: DataLoaderの実装
+ [x]: SiameseNetの実装
+ [x]: 学習フェーズの実装

# Code Structure
- **data_preprocessing.py**
  - データの前処理を行う
- **eda.py**
  - データのEDAを行う

## Reference 
・PyTorch transforms/Dataset/DataLoaderの基本動作を確認する 
https://qiita.com/takurooo/items/e4c91c5d78059f92e76d
・pytorch siamese-triplet
https://github.com/adambielski/siamese-triplet
・PyTorch/TorchVisionで複数の入力をモデルに渡したいケース
https://blog.shikoan.com/torchvision-multiple-inputs/
・pytorch siamese
https://vaaaaaanquish.hatenablog.com/entry/2019/02/23/214036


## Usage TensorBoard
```
ssh username@IP_Address -L 6006:localhost:6006
tensorboard --logdir <log_dir>

```
