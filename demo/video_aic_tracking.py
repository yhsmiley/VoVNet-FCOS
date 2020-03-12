# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
import copy
import numpy as np

from fcos_core.config import cfg
from predictor_aic import AICDemo
from tqdm import tqdm
from deep_sort.deepsort_tracker_BS import DeepSort as Tracker
from drawer import Drawer

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config_file",
        default="configs/aic/fcos_V_57_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="training_dir/aic/fcos_V_57_FPN_1x/model_<>.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min_image_size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show_mask_heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        '--video', 
        help='Path to video', 
        type=str
    )
    parser.add_argument(
        '--outputVideo', 
        help='Saves the output as a video file', 
        type=str, 
        default='out'
    )
    parser.add_argument(
        '--display', 
        action='store_true'
    )
    parser.add_argument(
        '--filename', 
        help='Name of file', 
        type=str,
        choices=['aic-FCOS-V-57-FPN-1x'],
        default='aic-FCOS-V-57-FPN-1x'
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights
    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
   
    if args.filename == 'aic-FCOS-V-57-FPN-1x':
        thresholds_for_classes = [0.5055431127548218, 0.48413702845573425, 0.4889211654663086, 0.3384720981121063, 0.5283428430557251, 0.45001116394996643, 0.3068044185638428, 0.48865219950675964, 0.33383917808532715, 0.5082104802131653, 0.35461583733558655, 0.4920685291290283, 0.3780040442943573, 0.47807830572128296, 0.3968450725078583, 0.37935689091682434, 0.4648495316505432, 0.49689367413520813, 0.40057891607284546, 0.4283473789691925]

    # prepare object that handles inference plus adds predictions on top of image
    aic_demo = AICDemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size,
    )

    vid_cap = cv2.VideoCapture(args.video)
    assert vid_cap.isOpened(), 'Video given invalid.'
    vid_width = int(vid_cap.get(3))
    vid_height = int(vid_cap.get(4))
    vid_fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total num frames: {}'.format(total_frames))

    out_vid = None
    if args.outputVideo:
        print('Outputing to video file')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if not os.path.exists('vids'):
            os.makedirs('vids')
        out_vid = cv2.VideoWriter(os.path.join(
            'vids', args.outputVideo+'_output.avi'), fourcc, vid_fps, (vid_width, vid_height))

    tracker = Tracker(max_age=30, nn_budget=70)
    print('DeepSORT Tracker inited!')
    drawer = Drawer()

    for frame_idx in tqdm(range(total_frames)):
        start_time = time.time()
        ret_val, frame = vid_cap.read()
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bbs = aic_demo.bbs_on_opencv_image(frame_RGB, classes=['person'])  # bbs in ltwh

        # initialize 0s embedding for testing
        features = []
        for i in range(len(bbs)):
            feature = np.ones((128,), dtype=np.float32)
            features.append(feature)
        # print(features[0].shape)

        tracks = tracker.update_tracks(bbs, features)

        if tracks is not None:
            show_frame = copy.deepcopy(frame)
            # drawer.draw_tracks(show_frame, tracks)
            drawer.draw_tracks_class(show_frame, tracks)
            if args.display:
                cv2.imshow("AIC detections", show_frame)
            if out_vid:
                out_vid.write(show_frame)

        if args.display:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if vid_cap:
        vid_cap.release()
    cv2.destroyAllWindows()
    if out_vid:
        out_vid.release()


if __name__ == "__main__":
    main()
