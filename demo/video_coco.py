# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os

from fcos_core.config import cfg
from predictor import COCODemo
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
        default="training_dir/fcos_V_57_FPN_1x/model_<>.pth",
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
    thresholds_for_classes = [0.48878026008605957, 0.5157854557037354, 0.5044764876365662, 0.49984481930732727, 0.5201855897903442, 0.5268604159355164, 0.542915403842926, 0.49880141019821167, 0.4862997531890869, 0.4703293442726135, 0.48309165239334106, 0.5980977416038513, 0.5839372873306274, 0.43605467677116394, 0.488334983587265, 0.5531968474388123, 0.6045324206352234, 0.5037994980812073, 0.5493407249450684, 0.5217279195785522, 0.5599054098129272, 0.541961133480072, 0.52456134557724, 0.590713620185852, 0.47225916385650635, 0.5106967091560364, 0.4527757167816162, 0.49706345796585083, 0.46771353483200073, 0.48096945881843567, 0.42433881759643555, 0.5834863781929016, 0.4681493937969208, 0.4610227644443512, 0.495819091796875, 0.49427446722984314, 0.5497299432754517, 0.49133870005607605, 0.5676629543304443, 0.4558263421058655, 0.4907543957233429, 0.5280252695083618, 0.4518597722053528, 0.4859154224395752, 0.4274418354034424, 0.5014893412590027, 0.44185423851013184, 0.49433377385139465, 0.5254765748977661, 0.49024903774261475, 0.47859156131744385, 0.5015891790390015, 0.4337356686592102, 0.5387176275253296, 0.45217376947402954, 0.49776363372802734, 0.46315333247184753, 0.5012166500091553, 0.46372056007385254, 0.4833259880542755, 0.4585243761539459, 0.55859375, 0.5683536529541016, 0.5178630352020264, 0.6591401100158691, 0.4779455363750458, 0.6092398762702942, 0.465324342250824, 0.5212925672531128, 0.543681800365448, 0.46491605043411255, 0.5281332731246948, 0.5284128189086914, 0.42081254720687866, 0.5965998768806458, 0.5190021991729736, 0.520578145980835, 0.5551347136497498, 0.3499980568885803, 0.5275787711143494]


    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
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
        composite = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        if args.display:
            cv2.imshow("COCO detections", composite)
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
