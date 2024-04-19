import os
import shutil
import json
import math
import cv2
import numpy as np
import scipy as sp
import pandas as pd
from datetime import datetime
import moviepy.editor as mpy
from argparse import ArgumentParser
from pose_features import load_eface, compute_features, FEATURES_LIST, STATS_LIST, EXPS_LIST
from vis_tools import segment_exps, plot_features, plot_scores, animate_features, create_demo


CLINIC_DATASET_MAPPING = {}
VIDEO_SIZE = (1080, 1920)
FPS = 30


def convert_time(time):
    if not type(time)==str:
        return 0
    pt = datetime.strptime(time,'%H:%M:%S')
    return pt.second + pt.minute*60 + pt.hour*3600

def convert_score(score):
    if np.isnan(score):
        return -1
    return int(score)

def create_clinic_dataset(source_dir, redcap_file, output_dir):
    data_dict = pd.read_csv(redcap_file).to_dict('records')
    meta_list = []
    for record in data_dict:
        isComplete = (record['Complete?']=='Complete' or record['Complete?']=='Unverified')
        isVideoFile = (record['Video File Name'].split(".")[-1].lower() in ['mp4', 'mov', 'm4v']) if type(record['Video File Name'])==str else False
        if isComplete and isVideoFile:
            # Extract file_path
            parent_dir = record['Immediate Folder Name']
            if parent_dir in CLINIC_DATASET_MAPPING:
                parent_dir = CLINIC_DATASET_MAPPING[parent_dir]
            file_name = record['Video File Name']
            if file_name in CLINIC_DATASET_MAPPING:
                file_name = CLINIC_DATASET_MAPPING[file_name]
            if '2017' in parent_dir:
                dir_path = os.path.join(source_dir, '2. Archives/2017 archive', parent_dir)
            elif '2018' in parent_dir:
                dir_path = os.path.join(source_dir, '2. Archives/2018 Archive', parent_dir)
            elif '2019' in parent_dir:
                dir_path = os.path.join(source_dir, '2. Archives/2019 Archive', parent_dir)
            elif '2020' in parent_dir:
                dir_path = os.path.join(source_dir, '2. Archives/2020 Archive', parent_dir)
            elif '2021' in parent_dir:
                dir_path = os.path.join(source_dir, '2. Archives/2021 Archive', parent_dir)
            elif '2022' in parent_dir:
                dir_path = os.path.join(source_dir, '2. Archives', parent_dir)
                if not os.path.exists(dir_path):
                    dir_path = os.path.join(source_dir, parent_dir)
            else:
                dir_path = os.path.join(source_dir, '1. Surgery', parent_dir)
                if not os.path.exists(dir_path):
                    dir_path = os.path.join(source_dir, '3. Clinic procedures', parent_dir)
            file_path = os.path.join(dir_path, file_name)
            if not os.path.exists(file_path):
                file_name = file_name.replace('/', ':')
                file_path = os.path.join(dir_path, file_name)

            # Generate video_id
            record_id = '{:08d}'.format(int(record['Record ID']))
            visit_num = '{:02d}'.format(int(record['Event Name'].replace('Visit', '').strip())) if 'Visit' in record['Event Name'] else '00'
            video_id = f'{record_id}-{visit_num}'
            case_dir = os.path.join(output_dir, video_id)
            os.makedirs(case_dir, exist_ok=True)
            print(video_id)
            
            # Write output video
            video_path = os.path.join(case_dir, video_id + '.mp4')
            video_clip = mpy.VideoFileClip(file_path)
            if not os.path.exists(video_path):
                clip_resized = video_clip.resize(VIDEO_SIZE)
                clip_resized.write_videofile(video_path, fps=FPS, logger=None)
                    
            # Save new_record
            metadata = {
                'Record ID': record_id, 
                'Visit Number': visit_num, 
                'Clinic Visit Date': record['clinic_visit_date'],
                'Grader': record['Grader'],
                'Facial Expressions': {
                    'Static': [convert_time(record['Start Time Static']), convert_time(record['End Time Static'])],
                    'Brow Elevation': [convert_time(record['Start Time Brow Elevation']), convert_time(record['End Time Brow Elevation'])],
                    'Eye Closure': [convert_time(record['Start Time Eye Closure']), convert_time(record['End Time Eye Closure'])],
                    'Smile': [convert_time(record['Start Time Smile (Best>Big>Gentle)']), convert_time(record['End Time Smile (Best>Big>Gentle)'])],
                    'EEEEE': [convert_time(record['Start Time E']), convert_time(record['End Time E'])],
                    'Pucker': [convert_time(record['Start Time Pucker']), convert_time(record['End Time Pucker'])],
                    'Say Words': [convert_time(record["Start Time Say 'Papa/Baby/Happy Birthday'"]), convert_time(record["End Time Say 'Papa/Baby/Happy Birthday'"])]
                },
                'Scores': {
                    'Static': {
                        'Brow at rest': convert_score(record['Brow at rest']),
                        'Palpebral fissure at rest': convert_score(record['Palpebral fissure at rest']),
                        'Nasolabial fold depth at rest': convert_score(record['Nasolabial fold depth at rest']),
                        'Oral commissure at rest': convert_score(record['Oral commissure at rest'])},
                    'Dynamic': {
                        'Brow elevation': convert_score(record['Brow elevation']),
                        'Gentle eye closure': convert_score(record['Gentle eye closure']),
                        'Full eye closure': convert_score(record['Full eye closure']),
                        'Nasolabial depth with smile': convert_score(record['Nasolabial depth with smile']),
                        'Nasolabial fold orientation with smile': convert_score(record['Nasolabial fold orientation with smile']),
                        'Oral commissure movement with smile': convert_score(record['Oral commissure movement with smile']),
                        'Lower lip movement with EEEEE': convert_score(record['Lower lip movement with EEEEE'])},
                    'Synkinesis': {
                        'Ocular': convert_score(record['Ocular']),
                        'Midfacial': convert_score(record['Midfacial']),
                        'Mentalis': convert_score(record['Mentalis']),
                        'Platysmal': convert_score(record['Platysmal'])},
                    'Composite': {
                        'Static domain score': convert_score(record['Static domain score']),
                        'Dynamic domain score': convert_score(record['Dynamic domain score']),
                        'Synkinesis domain score': convert_score(record['Synkinesis domain score']),
                        'Smile composite score': convert_score(record['Smile composite score'])}
                }
            }
            meta_file = os.path.join(case_dir, 'eface_' + video_id + '.json')
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            meta_list.append(metadata)

    dataframe = pd.DataFrame(meta_list)
    output_file = os.path.join(output_dir, 'dataset.csv')
    dataframe.to_csv(output_file, encoding='utf-8', index=False)

    return dataframe


def creat_meei_dataset(source_dir, redcap_file, output_dir):
    data_dict = pd.read_csv(redcap_file).to_dict('records')
    meta_list = []
    for record in data_dict:
        isComplete = (record['Complete?']=='Complete' or record['Complete?']=='Unverified')
        isVideoFile = (record['EXACT name of video '].split(".")[-1].lower() in ['mp4', 'mov', 'm4v']) if type(record['EXACT name of video '])==str else False
        if isComplete and isVideoFile:
            # Extract file_path
            record_id = record['Record ID']
            if 'Flaccid' in record_id:
                parent_dir = os.path.join('Flaccid', record_id[:-1], record_id)
            elif 'Synkinetic' in record_id:
                parent_dir = os.path.join('Synkinetic', record_id[11:-1] + 'Synkinetic', record_id)
                print(parent_dir)
            else:
                parent_dir = record_id
            file_name = record['EXACT name of video ']
            dir_path = os.path.join(source_dir, parent_dir)
            file_path = os.path.join(dir_path, file_name)

            # Generate video_id
            record_id = record['Record ID']
            video_id = record_id
            case_dir = os.path.join(output_dir, video_id)
            os.makedirs(case_dir, exist_ok=True)
            print(video_id)
            
            # Write output video
            video_path = os.path.join(case_dir, video_id + '.mp4')
            video_clip = mpy.VideoFileClip(file_path)
            if not os.path.exists(video_path):
                clip_resized = video_clip.resize(VIDEO_SIZE)
                clip_resized.write_videofile(video_path, fps=FPS, logger=None)
                    
            # Save new_record
            start, end = record['optimal static (rest) time period '].split('-')
            start_static = convert_time(start.strip())
            end_static = convert_time(end.strip())
            start, end = record['optimal brow elevation time period'].split('-')
            start_brow = convert_time(start.strip())
            end_brow = convert_time(end.strip())
            start, end = record['full eye closure optimal time period'].split('-')
            start_eye = convert_time(start.strip())
            end_eye = convert_time(end.strip())
            start, end = record['optimal smile time period'].split('-')
            start_smile = convert_time(start.strip())
            end_smile = convert_time(end.strip())
            start, end = record['optimal eeee time period'].split('-')
            start_eeee = convert_time(start.strip())
            end_eeee = convert_time(end.strip())
            metadata = {
                'Record ID': record_id, 
                'Grader': record['Grader (insert your name).1'],
                'Facial Expressions': {
                    'Static': [start_static, end_static],
                    'Brow Elevation': [start_brow, end_brow],
                    'Eye Closure': [start_eye, end_eye],
                    'Smile': [start_smile, end_smile],
                    'EEEEE': [start_eeee, end_eeee],
                    'Pucker': [0, 0],
                    'Say Words': [0, 0]
                },
                'Scores': {
                    'Static': {
                        'Brow at rest': convert_score(record['brows at rest .1']),
                        'Palpebral fissure at rest': convert_score(record['palpebral fissure at rest .1']),
                        'Nasolabial fold depth at rest': convert_score(record['nasolabial fold depth at rest.1']),
                        'Oral commissure at rest': convert_score(record['oral commissure at rest.1'])},
                    'Dynamic': {
                        'Brow elevation': convert_score(record['brow elevation with raising.1']),
                        'Gentle eye closure': convert_score(record['gentle eye closure.1']),
                        'Full eye closure': convert_score(record['full eye closure .1']),
                        'Nasolabial depth with smile': convert_score(record['nasolabial fold depth with smile.1']),
                        'Nasolabial fold orientation with smile': convert_score(record['nasolabial fold orientation with smile.1']),
                        'Oral commissure movement with smile': convert_score(record['oral commissure movement with smile .1']),
                        'Lower lip movement with EEEEE': convert_score(record['lower lip movement eeee .1'])},
                    'Synkinesis': {
                        'Ocular': convert_score(record['ocular synkinesis .1']),
                        'Midfacial': convert_score(record['midfacial synkinesis.1']),
                        'Mentalis': convert_score(record['mentalis synkinesis.1']),
                        'Platysmal': convert_score(record['platysmal synkinesis.1'])},
                    'Composite': {
                        'Static domain score': convert_score(record['static domain score.1']),
                        'Dynamic domain score': convert_score(record['dynamic domain score.1']),
                        'Synkinesis domain score': convert_score(record['Synkinesis domain score.1']),
                        'Smile composite score': convert_score(record['smile composite score.1'])}
                }
            }
            meta_file = os.path.join(case_dir, 'eface_' + video_id + '.json')
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            meta_list.append(metadata)

    dataframe = pd.DataFrame(meta_list)
    output_file = os.path.join(output_dir, 'dataset.csv')
    dataframe.to_csv(output_file, encoding='utf-8', index=False)
    return dataframe

def anonymise_dataset(data_root, clinic_dataset, meei_dataset):
    clinic_df = pd.read_csv(os.path.join(data_root, clinic_dataset, 'dataset.csv'))
    meei_df = pd.read_csv(os.path.join(data_root, meei_dataset, 'dataset.csv'))
    os.makedirs(os.path.join(data_root, photo_release), exist_ok=True)
    
    for record in clinic_df.to_records():
        record_id = record['Record ID']
        visit_num = record['Visit Number']
        if int(record_id) in photo_release_dict.values():
            source_id = '{:08d}-{:02d}'.format(record_id, visit_num)
            target_id = '{}Visit{}'.format(list(photo_release_dict.keys())[list(photo_release_dict.values()).index(record_id)], visit_num)
            source_dir = os.path.join(data_root, clinic_dataset, source_id)
            target_dir = os.path.join(data_root, photo_release, target_id)
            source_vid = os.path.join(source_dir, f'{source_id}.mp4')
            target_vid = os.path.join(target_dir, f'video_{target_id}.mp4')
            source_eface = os.path.join(source_dir, f'eface_{source_id}.json')
            target_eface = os.path.join(target_dir, f'eface_{target_id}.json')
            source_pose1 = os.path.join(source_dir, f'pose_{source_id}.npy')
            target_pose1 = os.path.join(target_dir, f'pose_{target_id}.npy')
            source_pose2 = os.path.join(source_dir, f'pose_{source_id}.mp4')
            target_pose2 = os.path.join(target_dir, f'pose_{target_id}.mp4')
            pose_data = np.load(source_pose1, allow_pickle=True).item()
            for key in pose_data:
                if not pose_data[key] and not pose_data[str(int(key)+1)]:
                    length = int(key) - 1
                    break
            length = min(length, len(pose_data))
            with open(source_eface) as f:
                eface_data = json.load(f)
            eface_data['Record ID'] = target_id.split('Visit')[0]
            eface_data.pop('Clinic Visit Date')
            eface_data.pop('Grader')
            print(f'Copying: {source_id} -> {target_id}')
            if not os.path.exists(target_pose1):
                os.makedirs(target_dir, exist_ok=True)
                video_clip = mpy.VideoFileClip(source_vid)
                video_clip = video_clip.subclip(0, length/30)
                video_clip.write_videofile(target_vid, fps=30, logger=None)
                with open(target_eface, 'w') as f:
                    json.dump(eface_data, f, indent=4)
                pose_clip = mpy.VideoFileClip(source_pose2)
                pose_clip = pose_clip.subclip(0, length/30)
                pose_clip.write_videofile(target_pose2, fps=30, logger=None)
                np.save(target_pose1, pose_data)
                print(f'Added: {target_id}, length {length/30:.2f}s')

        elif int(record_id) in anonymous_dict.values():
            source_id = '{:08d}-{:02d}'.format(record_id, visit_num)
            target_id = '{}Visit{}'.format(list(anonymous_dict.keys())[list(anonymous_dict.values()).index(record_id)], visit_num)
            source_dir = os.path.join(data_root, clinic_dataset, source_id)
            target_dir = os.path.join(data_root, anonymous, target_id)
            source_vid = os.path.join(source_dir, f'{source_id}.mp4')
            target_vid = os.path.join(target_dir, f'video_{target_id}.mp4')
            source_eface = os.path.join(source_dir, f'eface_{source_id}.json')
            target_eface = os.path.join(target_dir, f'eface_{target_id}.json')
            source_pose1 = os.path.join(source_dir, f'pose_{source_id}.npy')
            target_pose1 = os.path.join(target_dir, f'pose_{target_id}.npy')
            source_pose2 = os.path.join(source_dir, f'pose_{source_id}.mp4')
            target_pose2 = os.path.join(target_dir, f'pose_{target_id}.mp4')
            with open(source_eface) as f:
                eface_data = json.load(f)
            eface_data['Record ID'] = target_id.split('Visit')[0]
            eface_data.pop('Clinic Visit Date')
            eface_data.pop('Grader')
            print(f'Copying: {source_id} -> {target_id}')
            if True or not os.path.exists(target_vid):
                os.makedirs(target_dir, exist_ok=True)
                with open(target_eface, 'w') as f:
                    json.dump(eface_data, f, indent=4)
                shutil.copy2(source_pose2, target_pose2)
                shutil.copy2(source_pose1, target_pose1)
                print(f'Added: {target_id}, length {length/30:.2f} s')

    for record in meei_df.to_records():
        record_id = record['Record ID']
        source_id = record_id
        target_id = record_id
        source_dir = os.path.join(data_root, meei_dataset, source_id)
        target_dir = os.path.join(data_root, photo_release, target_id)
        source_vid = os.path.join(source_dir, f'{source_id}.mp4')
        target_vid = os.path.join(target_dir, f'video_{target_id}.mp4')
        source_eface = os.path.join(source_dir, f'eface_{source_id}.json')
        target_eface = os.path.join(target_dir, f'eface_{target_id}.json')
        source_pose1 = os.path.join(source_dir, f'pose_{source_id}.npy')
        target_pose1 = os.path.join(target_dir, f'pose_{target_id}.npy')
        source_pose2 = os.path.join(source_dir, f'pose_{source_id}.mp4')
        target_pose2 = os.path.join(target_dir, f'pose_{target_id}.mp4')
        if not os.path.exists(target_vid):
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(source_vid, target_vid)
            shutil.copy2(source_eface, target_eface)
            shutil.copy2(source_pose1, target_pose1)
            shutil.copy2(source_pose2, target_pose2)
            print('Added: {}'.format(target_vid))

