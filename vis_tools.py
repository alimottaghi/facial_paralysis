import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.animation import FuncAnimation
import moviepy.editor as mpy
from tqdm import tqdm
import seaborn as sns

from pose_features import FEATURES_LIST, STATS_LIST, EXPS_LIST

sns.set_theme(color_codes=True)


def segment_exps(eface, length, fps=30):
    segments = [(0, 0, None)]
    t = 0
    last_exp = None
    while t < length:
        exp_found = False
        for exp in EXPS_LIST:
            start, end = tuple(eface['Facial Expressions'][exp])
            if t >= start and t < end:
                exp_found = True
                if t == 0:
                    segments = [(0, 0, exp)]
                    last_exp = exp
                if exp != last_exp:
                    segments.append((t, t, exp))
                    last_exp = exp
                else:
                    segments[-1] = (segments[-1][0], t, exp)
                break
        if not exp_found:
            if last_exp is not None:
                segments.append((t, t, None))
                last_exp = None
            else:
                segments[-1] = (segments[-1][0], t, None)
        t += 1/fps
    return segments

def plot_landmarks(landmarks):
    plt.figure(figsize=(5.4, 9.6))
    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.xlim(0, 1080)
    plt.ylim(1920, 0)
    plt.show()

def plot_features(features, eface, out_path=None, show=False):
    fig, axs = plt.subplots(10, 1, sharex=True, figsize=(10, 35))
    fig.subplots_adjust(hspace=0)
    for i in range(10):
        axs[i].plot(np.arange(len(features[i]))/30, features[i])
        axs[i].set_ylabel(FEATURES_LIST[i])
        for j, exp in enumerate(EXPS_LIST):
            start, end = tuple(eface['Facial Expressions'][exp])
            axs[i].axvspan(xmin=start, xmax=end, facecolor=plt.get_cmap("tab10")(j), alpha=0.3)
    axs[9].set_xlabel('Time')
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()

def plot_scores(eface, out_path=None, show=False):
    fig, axs = plt.subplots(3, 1, figsize=(480/200, 1920/200), dpi=200)
    axs[0].bar(*zip(*eface['Scores']['Static'].items()))
    axs[0].axhline(y=50, color='tab:red', linewidth=1)
    axs[0].set_title('Static')
    axs[0].tick_params(axis='y', labelsize=6)
    axs[0].tick_params(axis='x', labelrotation=75, labelsize=6)
    axs[0].set_xticklabels(['Brow', 'Palpebral fissure', 'Nasolabial fold depth', 'Oral commissure'])
    axs[0].set_ylim(0, 100)
    dynamic_scores = eface['Scores']['Dynamic']
    dynamic_scores.pop("Nasolabial depth with smile", None)
    dynamic_scores.pop("Nasolabial fold orientation with smile", None)
    axs[1].bar(*zip(*eface['Scores']['Dynamic'].items()))
    axs[1].set_title('Dynamic')
    axs[1].tick_params(axis='y', labelsize=6)
    axs[1].tick_params(axis='x', labelrotation=75, labelsize=6)
    axs[1].set_xticklabels(['Brow elevation', 'Gentle eye closure', 'Full eye closure', 'Oral commissure', 'Lower lip'])
    axs[1].set_ylim(0, 100)
    axs[2].bar(*zip(*eface['Scores']['Synkinesis'].items()))
    axs[2].set_title('Synkinesis')
    axs[2].tick_params(axis='y', labelsize=6)
    axs[2].tick_params(axis='x', labelrotation=75, labelsize=6)
    axs[2].set_ylim(0, 100)
    plt.subplots_adjust(hspace=1)
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()

def animate_features(series1, series2, series3, segments, out_path=None, labels=None, fps=30):
    length = len(series1) / fps

    figure, axs = plt.subplots(3, 1, sharex=True, figsize=(1920/200, 1920/200), dpi=200)
    cmap = plt.get_cmap("tab10")
    axs[0].set_xlim(0, length)
    axs[0].set_ylim(-45, 45)
    axs[0].set_ylabel(labels[0])
    axs[1].set_xlim(0, length)
    axs[1].set_ylim(-45, 45)
    axs[1].set_ylabel(labels[1])
    axs[2].set_xlim(0, length)
    axs[2].set_ylim(-45, 45)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel(labels[2]) #, fontsize=10)
    plt1_segs = []
    plt2_segs = []
    plt3_segs = []
    for seg in segments:
        exp = seg[2]
        if exp is None:
            c = cmap(0)
        else:
            c = cmap(EXPS_LIST.index(exp) + 1)
        plt1, = axs[0].plot(seg[0], seg[1], linewidth=3, color=c)
        plt2, = axs[1].plot(seg[0], seg[1], linewidth=3, color=c)
        plt3, = axs[2].plot(seg[0], seg[1], linewidth=3, color=c)
        plt1_segs.append(plt1)
        plt2_segs.append(plt2)
        plt3_segs.append(plt3)

    x_segs = [[]*len(segments)]
    y1_segs = []
    y2_segs = []
    y3_segs = []
    for seg in segments:
        x_segs.append([])
        y1_segs.append([])
        y2_segs.append([])
        y3_segs.append([])
    pbar = tqdm(total=int(length * fps))
    def animation_function(t):
        pbar.update(1)
        i = int(t * fps)
        for j, seg in enumerate(segments):
            start, end = seg[0], seg[1]
            if start <= t < end:
                axs[0].set_title(seg[2])
                x_segs[j].append(t)
                y1_segs[j].append(series1[i])
                y2_segs[j].append(series2[i])
                y3_segs[j].append(series3[i])
                plt1_segs[j].set_xdata(x_segs[j])
                plt1_segs[j].set_ydata(y1_segs[j])
                plt2_segs[j].set_xdata(x_segs[j])
                plt2_segs[j].set_ydata(y2_segs[j])
                plt3_segs[j].set_xdata(x_segs[j])
                plt3_segs[j].set_ydata(y3_segs[j])
                return

    animation = FuncAnimation(figure,
                            func = animation_function,
                            frames = np.arange(0, int(length), 1/30), 
                            interval = 1000/30)

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    if out_path is not None:
        animation.save(out_path, writer=writer)

def create_demo(vis_file, ani_file, score_file, length, out_path=None):
    scores_img = mpy.ImageClip(score_file)
    vis_clip = mpy.VideoFileClip(vis_file)
    feat_clip = mpy.VideoFileClip(ani_file)
    score_clip = scores_img.set_duration(length)
    if out_path is not None:
        mpy.clips_array([[vis_clip, feat_clip, score_clip]]).write_videofile(out_path)
