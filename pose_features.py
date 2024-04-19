import os
import json
import math
import cv2
import numpy as np
import scipy as sp
import pandas as pd

# FEATURES_LIST = ['Bilateral lateral brow slope', 'Bilateral mid-brow slope',
#                  'Bilateral medial brow slope', 'Bilateral lateral canthal slope',
#                  'Bilateral oral commissure slope', 'Eyebrow height', 'Commissure symmetry',
#                  'Alar base symmetry', 'Mid-lower lip to mid-brow', 'Mid-lower lip to lateral canthus']
FEATURES_LIST = ['Canthal', 'Eye', 'Lateral Brow', 'Medial Brow', 'Commissure', 'Outer Mouth', 'Inner Mouth']
STATS_LIST = ['Max', 'Min', 'Range', 'Mean', 'Var', 'Skew', 'Kurtosis']
EXPS_LIST = ['Static', 'Brow Elevation', 'Eye Closure', 'Smile', 'EEEEE', 'Pucker', 'Say Words']
FPS = 30


def correct_tilt(landmarks):
    deg = 0
    for i in range(16):
        deg = deg + math.degrees(math.atan2(landmarks[32-i, 1]-landmarks[i, 1], landmarks[32-i, 0]-landmarks[i, 0]))
    tilt = deg / 16
    center = np.mean(landmarks[:32, :], axis=0)[:2]
    rotation = cv2.getRotationMatrix2D(center=center, angle=tilt, scale=1)
    for point in landmarks:
        point[:2] = np.dot(rotation, np.append(point[:2], 1))
    return landmarks

def slope(a, b):
    return (a[1]-b[1]) / (a[0]-b[0])

def slope_angle(a, b):
    return math.degrees(math.atan2(a[1]-b[1], a[0]-b[0]))

def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def distance1d(a, b):
    return math.sqrt((a-b)**2)

def perimeter(points_list):
    p = 0
    for i in range(len(points_list) - 1):
        p = p + distance(points_list[i], points_list[i+1])
    p = p + distance(points_list[0], points_list[-1])
    
    return p

def mdpi_features(landramks):
    A = distance1d(landramks[1][0], landramks[31][0])
    Bl = distance1d(landramks[60][0], landramks[64][0])
    Br = distance1d(landramks[68][0], landramks[72][0])
    C = distance(landramks[16], landramks[85])
    D = distance1d(landramks[1][0], landramks[60][0])
    E = distance1d(landramks[31][0], landramks[72][0])
    F = distance(landramks[60], landramks[85])
    G = distance(landramks[72], landramks[85])
    H = distance(landramks[60], landramks[55])
    I = distance(landramks[59], landramks[72])
    J = distance(landramks[55], landramks[85])
    K = distance(landramks[59], landramks[85])
    L = np.mean([landramks[i][1] for i in range(33, 37)])
    M = np.mean([landramks[i][1] for i in range(42, 46)])
    Nl = distance1d(landramks[61][1], landramks[67][1])
    Nr = distance1d(landramks[63][1], landramks[65][1])
    Ol = distance1d(landramks[69][1], landramks[75][1])
    Or = distance1d(landramks[71][1], landramks[73][1])
    N = np.mean([landramks[i][1] for i in range(60, 67)]) # (Nl + Nr)/2
    O = np.mean([landramks[i][1] for i in range(68, 75)]) # (Ol + Or)/2
    Pl = distance(landramks[77], landramks[87])
    Pu = distance(landramks[78], landramks[86])
    Ql = distance(landramks[81], landramks[83])
    Qu = distance(landramks[80], landramks[84])
    R = distance(landramks[36], landramks[85])
    S = distance(landramks[43], landramks[85])
    T = distance(landramks[35], landramks[85])
    U = distance(landramks[44], landramks[85])
    Vl = distance(landramks[76], landramks[85])
    Vr = distance(landramks[82], landramks[85])
    Wl = perimeter([landramks[76], landramks[77], landramks[78], landramks[79],
                   landramks[85], landramks[86], landramks[87]])
    Wr = perimeter([landramks[79], landramks[80], landramks[81], landramks[82],
                   landramks[83], landramks[84], landramks[85]])
    W = distance1d(landramks[76][0], landramks[82][0])
    X = distance(landramks[57], landramks[79])
    
    f0 = slope_angle(landramks[46], landramks[33])
    f1 = slope_angle(landramks[35], landramks[44])
    f2 = slope_angle(landramks[37], landramks[42])
    f3 = (M/L - 1) * 100
    f4 = slope(landramks[33], landramks[46])
    f5 = slope(landramks[35], landramks[44])
    f6 = slope(landramks[37], landramks[42])
    f7 = slope_angle(landramks[60], landramks[72])
    f8 = (Br/Bl - 1) * 100
    f9 = (E/D - 1) * 100
    f10 = (I/H - 1) * 100
    f11 = (O/N - 1) * 100
    f12 = (Or/Nl - 1) * 100
    f13 = (Ol/Nr - 1) * 100
    f14 = slope_angle(landramks[82], landramks[76])
    f15 = (G/F - 1) * 100
    f16 = (Ql/Pl - 1) * 100
    f17 = (Qu/Pu - 1) * 100
    f18 = max(Vl/A, Vr/A)
    f19 = max(Pl/W, Ql/W)
    f20 = max(Pu/W, Qu/W)
    f21 = max(Wl/W, Wr/W)
    f22 = slope_angle(landramks[59], landramks[55])
    f23 = slope_angle(landramks[57], landramks[85]) - 90
    f24 = (K/J - 1) * 100
    f24 = max(J/K, K/J)
    f25 = max(T/A, U/A)
    f26 = max(R/A, S/A)
    f27 = C/A
    f28 = X/A
    # features = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
    #     f10, f11, f12, f13, f14, f15, f16, f17, f18, f19,
    #     f20, f21, f22, f23, f24, f25, f26, f27, f28]
    features = [f0, f3, f10, f11, f14, f15, f16, f17, f22, f24]
    return features

def my_features(landmarks):
    f0 = slope_angle(landmarks[46], landmarks[33])  # bilateral lateral brow slope
    f1 = slope_angle(landmarks[44], landmarks[35])  # bilateral mid brow slope
    f2 = slope_angle(landmarks[42], landmarks[37])  # bilateral medial brow slope
    f3 = slope_angle(landmarks[72], landmarks[60])  # bilateral lateral canthal slope
    f4 = slope_angle(landmarks[82], landmarks[76])  # bilateral oral commissure slope
    f5 = (distance(landmarks[62], landmarks[35])/distance(landmarks[70], landmarks[44]) - 1) * 100  # mid brow to mid upper lid distance (eyebrow height)
    f6 = (distance(landmarks[76], landmarks[60])/distance(landmarks[82], landmarks[72]) - 1) * 100  # commissure to lateral canthus distance (commissure symmetry)
    f7 = (distance(landmarks[55], landmarks[60])/distance(landmarks[59], landmarks[72]) - 1) * 100  # alar base to lateral canthus distance  (alar base symmetry)
    f8 = (distance(landmarks[78], landmarks[60])/distance(landmarks[80], landmarks[72]) - 1) * 100  # mid lower lip to mid brow distance
    f9 = (distance(landmarks[86], landmarks[60])/distance(landmarks[84], landmarks[72]) - 1) * 100  # mid lower lip to lateral canthus distance
    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]

def my_stats(features):
    features_max = np.max(features, axis=0)
    features_min = np.min(features, axis=0)
    features_ptp = np.ptp(features, axis=0)
    features_mean = np.mean(features, axis=0)
    features_var = np.var(features, axis=0)
    features_skew = sp.stats.skew(features, axis=0)
    features_kurtosis = sp.stats.kurtosis(features, axis=0)
    return [features_max, features_min, features_ptp, features_mean, features_var, features_skew, features_kurtosis]

def load_eface(eface_file):
    with open(eface_file) as f:
        eface = json.load(f)
    return eface

def compute_features(pose_file):
    pose_data = np.load(pose_file, allow_pickle=True).item()
    features_list = []
    for frame in pose_data:
        frame_pose = pose_data.get(str(frame))
        if frame_pose:
            frame_landmarks = correct_tilt(frame_pose[0]['keypoints'])
            frame_features = my_features(frame_landmarks)
            features_list.append(frame_features)
        else:
            frame_features = [float("nan")] * len(FEATURES_LIST)
            features_list.append(frame_features)

    video_features = np.stack(features_list).T
    video_features_ma = np.zeros_like(video_features)
    for i in range(len(FEATURES_LIST)):
        video_features_ma[i] = np.convolve(video_features[i], np.ones(int(FPS/2)), 'same') / (FPS/2)
    return video_features_ma

def compute_stats(features, eface):
    stats_list = []
    for exp in EXPS_LIST:
        start, end = tuple(eface['Facial Expressions'][exp])
        if end > start:
            exp_features = features[:, start*FPS:end*FPS].T
            if exp_features.shape[0] == 0:
                exp_features = np.zeros_like(features).T
        else:
            exp_features = np.zeros_like(features).T
        exp_stats = my_stats(exp_features)
        stats_list.append(exp_stats)
    return stats_list


# New features
def canthal_metrics(keypoints):
    noise = (keypoints[55] + keypoints[56] + keypoints[57] + keypoints[58] + keypoints[59])/5
    left = distance(keypoints[60], noise)
    right = distance(keypoints[72], noise)
    return [left, right]

def eye_metrics(keypoints):
    left1 = distance(keypoints[61], keypoints[67])
    left2 = distance(keypoints[62], keypoints[66])
    left3 = distance(keypoints[63], keypoints[65])
    left = (left1 + left2 + left3)/3
    right1 = distance(keypoints[71], keypoints[73])
    right2 = distance(keypoints[70], keypoints[74])
    right3 = distance(keypoints[69], keypoints[75])
    right = (right1 + right2 + right3)/3
    return [left, right]

def lateral_brow_metrics(keypoints):
    noise = (keypoints[55] + keypoints[56] + keypoints[57] + keypoints[58] + keypoints[59])/5
    left1 = distance(keypoints[33], noise)
    left2 = distance(keypoints[34], noise)
    left3 = distance(keypoints[35], noise)
    left = (left1 + left2 + left3)/3
    right1 = distance(keypoints[46], noise)
    right2 = distance(keypoints[45], noise)
    right3 = distance(keypoints[44], noise)
    right = (right1 + right2 + right3)/3
    return [left, right]

def medial_brow_metrics(keypoints):
    noise = (keypoints[55] + keypoints[56] + keypoints[57] + keypoints[58] + keypoints[59])/5
    left1 = distance(keypoints[35], noise)
    left2 = distance(keypoints[36], noise)
    left3 = distance(keypoints[37], noise)
    left = (left1 + left2 + left3)/3
    right1 = distance(keypoints[44], noise)
    right2 = distance(keypoints[43], noise)
    right3 = distance(keypoints[42], noise)
    right = (right1 + right2 + right3)/3
    return [left, right]

def commissure_metrics(keypoints):
    noise = (keypoints[55] + keypoints[56] + keypoints[57] + keypoints[58] + keypoints[59])/5
    left1 = distance(keypoints[76], noise)
    left2 = distance(keypoints[88], noise)
    left = (left1 + left2)/2
    right1 = distance(keypoints[82], noise)
    right2 = distance(keypoints[92], noise)
    right = (right1 + right2)/2
    return [left, right]

def outer_mouth_metrics(keypoints):
    left1 = distance(keypoints[77], keypoints[87])
    left2 = distance(keypoints[78], keypoints[86])
    left = (left1 + left2)/2
    right1 = distance(keypoints[81], keypoints[83])
    right2 = distance(keypoints[80], keypoints[84])
    right = (right1 + right2)/2
    middle = distance(keypoints[79], keypoints[85])
    return [left, right]

def inner_mouth_metrics(keypoints):
    left = distance(keypoints[89], keypoints[95])
    right = distance(keypoints[91], keypoints[93])
    middle = distance(keypoints[90], keypoints[94])
    return [left, right]

def extract_features(pose, save_path=None):
    features_list = []
    for frame in pose:
        frame_pose = pose.get(str(frame))
        if frame_pose:
            keypoints = frame_pose[0]['keypoints']
            features_list.append(canthal_metrics(keypoints) + eye_metrics(keypoints) + lateral_brow_metrics(keypoints) + medial_brow_metrics(keypoints) + commissure_metrics(keypoints) + outer_mouth_metrics(keypoints) + inner_mouth_metrics(keypoints))
        else:
            frame_features = [float("nan")] * 14
            features_list.append(frame_features)

    # Run moving average
    video_features = np.stack(features_list).T
    video_features_ma = np.zeros_like(video_features)
    for i in range(14):
        video_features_ma[i] = np.convolve(video_features[i], np.ones(int(30/2)), 'same') / (30/2)
    canthal_feat = video_features_ma[0:2]
    eye_feat = video_features_ma[2:4]
    lateral_brow_feat = video_features_ma[4:6]
    medial_brow_feat = video_features_ma[6:8]
    commissure_feat = video_features_ma[8:10]
    outer_mouth_feat = video_features_ma[10:12]
    inner_mouth_feat = video_features_ma[12:14]

    # Normalize
    canthal_range = np.ptp(np.nan_to_num(canthal_feat), axis=1).max()
    canthal_feat = (canthal_feat[0] - canthal_feat[1]) / canthal_range
    eye_range = np.ptp(np.nan_to_num(eye_feat), axis=1).max()
    eye_feat = (eye_feat[0] - eye_feat[1]) / eye_range
    lateral_brow_range = np.ptp(np.nan_to_num(lateral_brow_feat), axis=1).max()
    lateral_brow_feat = (lateral_brow_feat[0] - lateral_brow_feat[1]) / lateral_brow_range
    medial_brow_range = np.ptp(np.nan_to_num(medial_brow_feat), axis=1).max()
    medial_brow_feat = (medial_brow_feat[0] - medial_brow_feat[1]) / medial_brow_range
    commissure_range = np.ptp(np.nan_to_num(commissure_feat), axis=1).max()
    commissure_feat = (commissure_feat[0] - commissure_feat[1]) / commissure_range
    outer_mouth_range = np.ptp(np.nan_to_num(outer_mouth_feat), axis=1).max()
    outer_mouth_feat = (outer_mouth_feat[0] - outer_mouth_feat[1]) / outer_mouth_range
    inner_mouth_range = np.ptp(np.nan_to_num(inner_mouth_feat), axis=1).max()
    inner_mouth_feat = (inner_mouth_feat[0] - inner_mouth_feat[1]) / inner_mouth_range
    extracted_features = np.stack([canthal_feat, eye_feat, lateral_brow_feat, medial_brow_feat, commissure_feat, outer_mouth_feat, inner_mouth_feat])
    if save_path:
        np.save(save_path, extracted_features)
    return extracted_features