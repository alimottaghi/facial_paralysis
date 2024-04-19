import os
import json
import numpy as np
import scipy as sp
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import seaborn as sns

from pose_features import load_eface, compute_features, compute_stats, FEATURES_LIST, STATS_LIST, EXPS_LIST

CORR_LIST = [
    ('Static', ('Static', 'Brow at rest')),
    ('Static', ('Static', 'Palpebral fissure at rest')),
    ('Static', ('Static', 'Nasolabial fold depth at rest')),
    ('Static', ('Static', 'Oral commissure at rest')),
    ('Brow Elevation', ('Dynamic', 'Brow elevation')),
    ('Eye Closure', ('Dynamic', 'Gentle eye closure')),
    ('Eye Closure', ('Dynamic', 'Full eye closure')),
    ('Smile', ('Dynamic', 'Nasolabial depth with smile')),
    ('Smile', ('Dynamic', 'Nasolabial fold orientation with smile')),
    ('Smile', ('Dynamic', 'Oral commissure movement with smile')),
    ('EEEEE', ('Dynamic', 'Lower lip movement with EEEEE')),
]


def extract_data(root_dir, case_ids=None, output_file=None):
    case_ids = case_ids if len(case_ids) > 0 else sorted(os.listdir(root_dir))
    output_file = os.path.join(root_dir, 'extract_data.npy') if output_file is None else output_file
    if os.path.exists(output_file):
        extracted_data = np.load(output_file, allow_pickle=True).item()
        return extracted_data
    extracted_data = {}
    for case_id in case_ids:
        case_dir = os.path.join(root_dir, case_id)
        if not os.path.isdir(case_dir):
            continue
        print(case_id)
        pose_file = os.path.join(root_dir, case_id, 'pose_' + case_id + '.npy')
        eface_file = os.path.join(root_dir, case_id, 'eface_' + case_id + '.json')
        eface = load_eface(eface_file)
        features = compute_features(pose_file)
        stats = compute_stats(features, eface)
        extracted_data[case_id] = {'stats': stats, 'scores': eface['Scores']}
    np.save(output_file, extracted_data)
    return extracted_data

def plot_correlations(extracted_data, output_dir='plots', corr_list=CORR_LIST):
    for pair in corr_list:
        print(pair)
        cur_exp = pair[0]
        cur_eface = pair[1]
        features = []
        scores = np.array([extracted_data[case_id]['scores'][cur_eface[0]][cur_eface[1]] for case_id in extracted_data])
        r_values = np.zeros((len(STATS_LIST), len(FEATURES_LIST)))
        p_values = np.zeros((len(STATS_LIST), len(FEATURES_LIST)))
        fig, axes = plt.subplots(len(STATS_LIST), len(FEATURES_LIST), figsize=(20000/200, 14000/200), dpi=200)
        for i in range(len(STATS_LIST)):
            for j in range(len(FEATURES_LIST)):
                stat_ij = np.array([extracted_data[case_id]['stats'][EXPS_LIST.index(cur_exp)][i][j] for case_id in extracted_data])
                stat_ij[np.isnan(stat_ij)] = 0
                features.append(stat_ij)
                sns.regplot(x=stat_ij, y=scores, ax=axes[i, j])
                r, p = sp.stats.pearsonr(stat_ij, scores)
                axes[i, j].set_title('r={:.2f}, p={:.2g}'.format(r, p))
                r_values[i, j], p_values[i, j] = r, p
        for i in range(len(STATS_LIST)):
            axes[i, 0].set_ylabel(STATS_LIST[i])
        for j in range(len(FEATURES_LIST)):
            axes[-1, j].set_xlabel(FEATURES_LIST[j])
        fig.suptitle = f'{cur_exp}-{cur_eface[1]} - R:{np.unravel_index(r_values.argmin(), r_values.shape)} P:{np.unravel_index(p_values.argmin(), p_values.shape)}'
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, 'Correlation {}-{}.png'.format(cur_exp, cur_eface[1]))
        fig.savefig(fig_path, bbox_inches='tight', pad_inches=0, dpi=100)
    
def plot_regressions(extracted_data, alpha=10, output_dir='plots', corr_list=CORR_LIST):
    for pair in corr_list:
        print(pair)
        cur_exp = pair[0]
        cur_eface = pair[1]
        features = np.zeros((len(extracted_data), len(STATS_LIST) * len(FEATURES_LIST)))
        scores = np.array([extracted_data[case_id]['scores'][cur_eface[0]][cur_eface[1]] for case_id in extracted_data])
        for k, case_id in enumerate(extracted_data):
            for i in range(len(STATS_LIST)):
                for j in range(len(FEATURES_LIST)):
                    features[k, i * len(FEATURES_LIST) + j] = extracted_data[case_id]['stats'][EXPS_LIST.index(cur_exp)][i][j]
        features[np.isnan(features)] = 0
        X_train, X_test, y_train, y_test = train_test_split(features, scores, test_size=0.3, random_state=0)
        ols = linear_model.Lasso(alpha=alpha)
        model = ols.fit(X_train, y_train)
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
        y_hat_test[y_hat_test < 0] = 0
        y_hat_test[y_hat_test > 100] = 100
        train_rmse = np.sqrt(mean_squared_error(y_train, y_hat_train))
        val_rmse = np.sqrt(mean_squared_error(y_test, y_hat_test))
        print(f'Train RMSE: {train_rmse}')
        print(f'Val RMSE: {val_rmse}')
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train, y_hat_train, label='Train')
        plt.scatter(y_test, y_hat_test, label='Val')
        plt.xlabel('Ground Truth')
        plt.ylabel('Predicted')
        plt.title(f'{cur_exp}-{cur_eface[1]} - RMSE: {val_rmse:.2f}')
        plt.legend()
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, 'Regression {}-{}.png'.format(cur_exp, cur_eface[1]))
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0, dpi=100)

def create_clip_dataset(root_dir, case_ids=None, output_dir='splits', corr_list=CORR_LIST, reset=False):
    case_ids = case_ids if len(case_ids) > 0 else sorted(os.listdir(root_dir))
    for pair in corr_list:
        print(pair)
        cur_exp = pair[0]
        cur_eface = pair[1]
        output_pre = cur_exp.split(' ')[0].lower()
        labels = []
        for case in case_ids:
            case_dir = os.path.join(root_dir, case)
            if not os.path.isdir(case_dir): continue
            video_path = os.path.join(case_dir, f'pose_{case}.mp4')
            eface_path = os.path.join(case_dir, f'eface_{case}.json')
            output_path = os.path.join(case_dir, f'{output_pre}_{case}.mp4')
            eface_data = json.load(open(eface_path))
            exp_time = eface_data['Facial Expressions'][cur_exp]
            exp_score = int(eface_data['Scores'][cur_eface[0]][cur_eface[1]]) // 20
            if exp_score==5: exp_score=4
            print(f'{case}: {exp_time} - {exp_score}')
            if reset or not os.path.exists(output_path):
                clip = mpy.VideoFileClip(video_path)
                if exp_time[1]<clip.duration and exp_time[1]>exp_time[0]:
                    clip.subclip(exp_time[0], exp_time[1]).write_videofile(output_path, logger=None)
                    labels.append([os.path.join(case, f'{output_pre}_{case}.mp4'), exp_score])
            else:
                labels.append([os.path.join(case, f'{output_pre}_{case}.mp4'), exp_score])

        dataset_annot= '\n'.join([' '.join([str(x) for x in row]) for row in labels])
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{cur_eface[1]} - {cur_exp}.csv'), 'w') as f:
            f.write(dataset_annot)
