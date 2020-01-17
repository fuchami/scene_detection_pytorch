#!/bin/bash
source activate tf-gpu

python main.py -i True -a False -t False -s True --normalize False --merge concat
python main.py -i True -a True -t False -s True --normalize False --merge concat
python main.py -i True -a False -t True -s True --normalize False --merge concat
python main.py -i True -a True -t True -s True --normalize False --merge concat

python main.py -i True -a False -t False -s True --normalize True --merge concat
python main.py -i True -a True -t False -s True --normalize True --merge concat
python main.py -i True -a False -t True -s True --normalize True --merge concat
python main.py -i True -a True -t True -s True --normalize True --merge concat

python main.py -i True -a True -t True -s True --normalize False --merge concat --model siamese
python main.py -i True -a True -t True -s True --normalize True --merge concat --model siamese
