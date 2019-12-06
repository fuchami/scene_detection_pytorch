# coding:utf-8
"""
BBC Planet Earth Dataset
make Multimodal Labeling file

[x] 画像: 各ショットのミドルフレーム
[ ] テキスト: ショットの中心のフレームから20sec間のテキストを埋め込む
[x] 音声： ショットの中心のフレームから10sec間の音声を抽出
[x] timestamp: ショットの開始終了時間、ショットの長さ

[shot_id, scene_id, start_frame, end_frame, image, audio, text, start_time, end_time, shot_length]
"""

#%% 
import os,sys,glob,csv
from pydub import AudioSegment
from pydub.silence import split_on_silence
import subprocess
import numpy as np
import pandas as pd

#%% dump txt file
movie_path_list = sorted(glob.glob("./BBC_Planet_Earth_Dataset/src/*.mp4"))
save_dir = './BBC_Planet_Earth_Dataset/text/src/'
if not os.path.exists(save_dir): os.makedirs(save_dir) 

for movie_path in movie_path_list:
    movie_name, _ = os.path.splitext(os.path.basename(movie_path))

    # dump text ssa file
    cmd = f'ffmpeg -i {movie_path} -map 0:4 {save_dir}{movie_name}.ssa'
    print(cmd)
    subprocess.call(cmd, shell=True)
    # ssa -> txt
    cmd = f'mv {save_dir}{movie_name}.ssa {save_dir}{movie_name}.txt'
    print(cmd)
    subprocess.call(cmd, shell=True)

#%% txtファイルの整形
txt_path_list = sorted(glob.glob("./BBC_Planet_Earth_Dataset/text/src/*"))
save_dir = "./BBC_Planet_Earth_Dataset/text/"

for txt_path in txt_path_list:
    txt_name, _ = os.path.splitext(os.path.basename(txt_path))
    with open(txt_path) as f:
        txt = [s.split(',') for s in f.readlines()]
        txt = txt[12:]
    
    # write
    with open(f'{save_dir}{txt_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end','start_sec', 'end_sec', 'text'])
        for t in txt:
            text = ''.join(t[9:]).rstrip().replace('\\', ' ')
            # 0:00:30.12 → second

            h,m,s = t[1].split(':')
            start_sec = round(int(m)*60 + float(s),2)
            h,m,s = t[2].split(':')
            end_sec = round(int(m)*60 + float(s),2)
            row = [t[1], t[2], start_sec, end_sec, text]
            writer.writerow(row)

#%% dump wav file
movie_path_list = sorted(glob.glob("./BBC_Planet_Earth_Dataset/src/*.mp4"))
save_dir = './BBC_Planet_Earth_Dataset/audio/src/'
if not os.path.exists(save_dir): os.makedirs(save_dir) 
for movie_path in movie_path_list:
    movie_name, _ = os.path.splitext(os.path.basename(movie_path))
    cmd = "ffmpeg -i "+movie_path+" -map 0:2 -codec:a copy "+save_dir+movie_name+".m4a"
    print(cmd)
    subprocess.call(cmd, shell=True)

    # m4a -> wav
    cmd = "ffmpeg -i "+save_dir+movie_name+".m4a "+save_dir+movie_name+".wav"
    subprocess.call(cmd, shell=True)

#%%
class MultimodalData(object):
    def __init__(self,annotator,movie_name, image=True, audio=True, text=False):
        self.image = image
        self.audio = audio
        self.test = text
        self.fps = 25
        self.episode = movie_name.split('_', 1)[0] # 01~11の値
        print(self.episode)

        annotator = annotator # 0~4の値
        shot_dir = './BBC_Planet_Earth_Dataset/annotations/shots/'
        scene_dir = './BBC_Planet_Earth_Dataset/annotations/scenes/'
        save_dir = f'./BBC_Planet_Earth_Dataset/dataset/annotator_{annotator}/'
        audio_path = f'./BBC_Planet_Earth_Dataset/audio/src/bbc_{self.episode}.wav'

        self.shot_id     = []
        self.scene_id    = []
        self.start_frame = []
        self.end_frame   = []
        self.start_sec   = []
        self.end_sec     = []
        self.middle_sec  = []
        self.shot_sec    = []
        self.image_data  = []
        self.audio_data  = []
        self.text_data   = []

        """ load shot txt """
        shot_txt_path = glob.glob(f'{shot_dir}{movie_name}*')[0]
        print(shot_txt_path)
        self.shot_data = self.load_shot_txt(shot_txt_path)

        """ load scene txt """
        scene_txt_path = glob.glob(f'{scene_dir}annotator_{annotator}/{movie_name}*')[0]
        print(scene_txt_path)
        self.scene_data = self.load_scene_txt(scene_txt_path)

        """ dump shot audio file """
        if audio: self.sound = AudioSegment.from_file(audio_path, format="wav")

        """ make dataset """
        self.make_dataset()

        """ dump """
        dataset = pd.DataFrame({
            "shot_id":self.shot_id,
            "scene_id":self.scene_id,
            "start_frame":self.start_frame,
            "end_frame":self.end_frame,
            "image":self.image_data,
            "audio":self.audio_data,
            "start_sec":self.start_sec,
            "end_sec":self.end_sec,
            "middle_sec":self.middle_sec,
            "shot_sec":self.shot_sec
        })

        if not os.path.exists(save_dir): os.makedirs(save_dir)
        dataset.to_csv(f'{save_dir}{movie_name}.csv')

    def load_shot_txt(self, path):
        with open(path, 'r') as f:
            data = f.readlines()
        data = [d.replace('\n', '') for d in data] # 改行を削除
        data = [d.split('\t') for d in data] # \tでリストを分割
        for i in range(len(data)):
            data[i] = [int(j) for j in data[i]] # int型へ
        return data
    
    def load_scene_txt(self, path):
        with open(path, 'r') as f:
            data = f.readlines()
        return data[0].split(',')
    
    def make_dataset(self):
        shot_id = 0
        scene_id = -1

        for num_shot in self.shot_data:
            if str(shot_id) in self.scene_data: scene_id += 1

            self.shot_id.append(shot_id)
            self.scene_id.append(scene_id)
            self.start_frame.append(num_shot[0])
            self.end_frame.append(num_shot[1])

            """ cals start/end/middle sec and frame """
            start_sec = float(num_shot[0]/self.fps)
            end_sec = float(num_shot[1]/self.fps)
            shot_sec = end_sec - start_sec
            shot_sec = round(shot_sec, 2) # 小数点2桁に丸め込み
            middle_sec = (end_sec - start_sec)/2 + start_sec
            middle_sec = round(middle_sec, 2) # 小数点2桁に丸め込み
            middle_frame = int((num_shot[1]-num_shot[0])/2) + num_shot[0]

            self.start_sec.append(start_sec)
            self.end_sec.append(end_sec)
            self.middle_sec.append(middle_sec)
            self.shot_sec.append(shot_sec)

            if self.image: self.image_data.append(f'./BBC_Planet_Earth_Dataset/frame/bbc_{self.episode}/{middle_frame}.jpg')
            if self.audio: self.audio_data.append(self.dub_audio(shot_id,middle_sec*1000))
            """ TODO: text datat"""

            shot_id += 1
    
    def dub_audio(self, shot_id, middle_ms, ms_width=5000):
        audio_dir = f'./BBC_Planet_Earth_Dataset/audio/bbc_{self.episode}/'
        if not os.path.exists(audio_dir): os.makedirs(audio_dir)
        audio_path = f'{audio_dir}{shot_id}.wav'

        shot_sound = self.sound[middle_ms-ms_width:middle_ms+ms_width]
        if not os.path.exists(audio_path): shot_sound.export(audio_path+'', format="wav")
        return audio_path

#%%
movie_name_list = [os.path.basename(i) for i in sorted(glob.glob('./BBC_Planet_Earth_Dataset/annotations/shots/*'))]
annotators_list = [str(i)  for i in range(5)]

for annotator in annotators_list:
    for movie_name in movie_name_list:
        movie_name,_ = os.path.splitext(movie_name)
        multimodaldata = MultimodalData(annotator, movie_name)

#%%
