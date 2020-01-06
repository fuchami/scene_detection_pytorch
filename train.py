# coding:utf-8
import os,sys
import subprocess

def main():
    models = ['siamese', 'triplet']
    margins = [0.1, 0.2, 0.5, 1.]
    normalize = ['False', 'True']
    merge = ['concat', 'mcb']
    weight = ['place', 'imagenet']

    for w in weight:
        for m in merge:
            for margin in margins:
                for norm in normalize:
                    for model in models:
                        cmd = ['python', 'main.py', '--model', model, '--margin', margin,
                                '-normalize', norm, '--merge', m, '--weight', w]
                        print(cmd)
                        result = subprocess.check_output(cmd)
                        print(result)

if __name__ == "__main__":
    main()