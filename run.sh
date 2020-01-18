#!/bin/bash
python main.py -i True -a True -t True -s True --merge concat --model triplet
python main.py -i True -a True -t True -s True --merge mcb --model triplet

python main.py -i True -a True -t True -s True --merge concat --model siamese

python main.py -i True -a False -t False -s True --merge concat --model triplet
python main.py -i True -a True -t False -s True --merge concat --model triplet
python main.py -i True -a False -t True -s True --merge concat --model triplet
python main.py -i True -a True -t True -s True --merge concat --model triplet

