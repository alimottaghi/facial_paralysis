import os
from argparse import ArgumentParser
from pose_features import load_eface, extract_features
from vis_tools import segment_exps, plot_features, plot_scores, animate_features, create_demo, FEATURES_LIST
from fp_dataset import FPS
from learning import extract_data, plot_correlations, plot_regressions, create_clip_dataset, CORR_LIST


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='/pasteur/u/mottaghi/datasets/facial_paralysis', help='Dataset Root Directory')
    parser.add_argument('--case-ids', nargs='+', default=[])
    args = parser.parse_args()

    root_dir = args.root_dir
    case_ids = args.case_ids if len(args.case_ids) > 0 else sorted(os.listdir(args.root_dir))
    case_ids = [case_id for case_id in case_ids if 'Patient' in case_id]
    for case_id in case_ids:
        print(case_id)
        eface_file = os.path.join(root_dir, case_id, f'eface_{case_id}.json')
        eface = load_eface(eface_file)
        pose_file = os.path.join(root_dir, case_id, f'pose_{case_id}.npy')
        features = extract_features(pose_file)
        length = len(features[0]) / FPS
        segments = segment_exps(eface, length)

        feat_file = os.path.join(root_dir, case_id, f'feat_{case_id}.jpg')
        if not os.path.exists(feat_file):
            plot_features(features, eface, out_path=feat_file, show=False)

        score_file = os.path.join(root_dir, case_id, f'score_{case_id}.jpg')
        if not os.path.exists(score_file):
            plot_scores(eface, out_path=score_file, show=False)

        ani_file = os.path.join(root_dir, case_id, f'feat_{case_id}.mp4')
        if not os.path.exists(ani_file):
            animate_features(features[5], features[6], features[7], segments, out_path=ani_file, labels=[FEATURES_LIST[5], FEATURES_LIST[6], FEATURES_LIST[7]])

        if os.path.exists(os.path.join(root_dir, case_id, f'vis_{case_id}.mp4')):
            vis_file = os.path.join(root_dir, case_id, f'vis_{case_id}.mp4')
        else:
            vis_file = os.path.join(root_dir, case_id, f'pose_{case_id}.mp4')
        demo_file = os.path.join(root_dir, case_id, f'demo_{case_id}.mp4')
        if not os.path.exists(demo_file):
            create_demo(vis_file, ani_file, score_file, length, out_path=demo_file)

    extracted_data = extract_data(root_dir, case_ids, output_file='clinic/extracted_clinic.npy')
    plot_correlations(extracted_data, output_dir='clinic')
    plot_regressions(extracted_data, output_dir='clinic')

    # create_clip_dataset(root_dir, case_ids, output_dir='clinic')


