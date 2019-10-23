"""
動画から1秒単位でフレーム切り出し
"""

import cv2
import numpy as np
import glob
import moviepy.editor as mp
import wave
import struct
from scipy import fromstring, int16

import math
import os, sys, glob
import argparse

class MovieSplit(object):

    def __init__(self, video_path, save_dir, between=1):
        """ Usage
        video_path:分割したい動画ファイル名
        save_dir: 分割したフレームの保存先
        between: 分割する間隔。そのままフレーム分割するのであれば1
        """

        self.video_path = video_path
        self.between = between
        print(video_path)
        basename, ext = os.path.splitext(os.path.basename(self.video_path))
        self.save2frame =  save_dir + '/' + basename + '/'
        if not os.path.exists(self.save2frame):
            os.makedirs(self.save2frame)

    """ movie to frame """
    def mv2frame(self):
        print("split video to frame images")
        cnt = 0

        # load movie
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("FRAME_COUNT", frame_cnt)
        print("fps: ", fps)
        print("between: ", self.between)

        print('--- video2frame start ---')
        for idx in range(0, frame_cnt, self.between):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            savepath_name = self.save2frame + str(cnt) + ".jpg"
            print('frame write:', savepath_name)
            cv2.imwrite(savepath_name, cap.read()[1])

            cnt += 1

        print("--- done. ---")
        cap.release()


def main(args):
    src_path_list = glob.glob(args.srcpath + '*.mp4')
    for src_path in src_path_list:
        print('lets :', src_path)
        mvsplit = MovieSplit(src_path, args.save2frame)
        mvsplit.mv2frame()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='movie convert to frame & audio files')
    parser.add_argument('--srcpath', '-p', default='./BBC_Planet_Earth_Dataset/src/', type=str)
    parser.add_argument('--save2frame', '-f',default='./BBC_Planet_Earth_Dataset/frame/', type=str)
    parser.add_argument('--save2audio', '-a', type=str)
    parser.add_argument('--between', '-b', type=int, default=1)
    parser.add_argument('--mode', '-m')

    args = parser.parse_args()
    main(args)