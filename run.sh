#!/bin/bash

# マルチモーダルの比較
python main.py --model triplet --image --merge concat
python main.py --model triplet --image --audio --merge concat
python main.py --model triplet --image --text --merge concat
python main.py --model triplet --image --time --merge concat

python main.py --model triplet --image --audio --text --merge concat
python main.py --model triplet --image --audio --text --time --merge concat

# out unitの比較
python main.py --model triplet --image --audio --text --time --outdim 64 --merge concat

# mcbの導入
python main.py --model triplet --image --audio --text --time --merge mcb
