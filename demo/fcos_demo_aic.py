# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os

from fcos_core.config import cfg
from predictor_aic import AICDemo

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
        "--images_dir",
        default="demo/images",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--min_image_size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
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

    demo_im_names = os.listdir(args.images_dir)

    # prepare object that handles inference plus adds predictions on top of image
    aic_demo = AICDemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    for im_name in demo_im_names:
        img = cv2.imread(os.path.join(args.images_dir, im_name))
        if img is None:
            continue
        start_time = time.time()
        composite = aic_demo.run_on_opencv_image(img)
        print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
        cv2.imshow(im_name, composite)
    print("Press any key to exit ...")
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
        exit()

if __name__ == "__main__":
    main()

