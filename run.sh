#!/bin/bash
# audioをmaxpoolに変えました

#python main.py --model triplet --image --audio --text --time --merge mcb 

# SOTA 全部入りのconcat
python main.py --model triplet --image --audio --text --time --merge concat 

python main.py --model triplet --image --audio --time --merge concat --epoch 100
# python main.py --model triplet --image --audio --text --time --merge max 

# 比較用siamese DONE!
# python main.py --model siamese --image --audio --text --time --merge concat

# mcbの導入
# python main.py --model triplet --image --audio --text --time --merge mcb
# python main.py --model triplet --image --audio --text --time --merge mcb --margin 1.
# python main.py --model triplet --image --audio --text --time --merge mcb --margin 0.5


# # マルチモーダルの比較
# python main.py --model triplet --image --merge concat
# python main.py --model triplet --image --audio --merge concat #これでコケル
# python main.py --model triplet --image --text --merge concat
# python main.py --model triplet --image --time --merge concat
# python main.py --model triplet --image --audio --text --merge concat




