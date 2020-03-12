# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os

from fcos_core.config import cfg
from predictor_aic import AICDemo
from tqdm import tqdm

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config_file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
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
    thresholds_for_classes = [0.46691036224365234, 0.4386535882949829, 0.4971639811992645, 0.34772348403930664, 0.4150613844394684, 0.43036216497421265, 0.23884204030036926, 0.4007486402988434, 0.2018958032131195, 0.4679737091064453, 0.29823338985443115, 0.44161197543144226, 0.3587378263473511, 0.4934690296649933, 0.36115896701812744, 0.332570344209671, 0.29670703411102295, 0.45676225423812866, 0.2644682824611664, 0.41843223571777344]

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

    for frame_idx in tqdm(range(total_frames)):
        start_time = time.time()
        ret_val, img = vid_cap.read()
        composite = aic_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        if args.display:
            cv2.imshow("AIC detections", composite)
        if out_vid:
            out_vid.write(composite)
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
