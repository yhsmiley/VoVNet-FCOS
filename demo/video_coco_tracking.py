# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
import copy
import numpy as np

from fcos_core.config import cfg
from predictor_coco import COCODemo
from tqdm import tqdm
from deep_sort.deepsort_tracker_BS import DeepSort as Tracker
from drawer import Drawer

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config_file",
        default="configs/vovnet/fcos_V_57_FPN_2x.yaml",
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
        '--filename', 
        help='Name of file', 
        type=str,
        choices=['FCOS-V-39-FPN-1x', 'FCOS-V-57-MS-FPN-2x','FCOS-V-93-MS-FPN-2x'],
        default='FCOS-V-57-MS-FPN-2x'
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
    if args.filename == 'FCOS-V-39-FPN-1x':
        thresholds_for_classes = [0.4871613681316376, 0.45935359597206116, 0.47524023056030273, 0.46535882353782654, 0.5457630753517151, 0.5112412571907043, 0.5289180278778076, 0.4660639464855194, 0.45085862278938293, 0.47864195704460144, 0.4975494146347046, 0.4788793921470642, 0.5068866014480591, 0.4542357325553894, 0.4678514003753662, 0.4941931366920471, 0.5626294612884521, 0.5301299095153809, 0.5370022058486938, 0.5048402547836304, 0.4960610270500183, 0.6174976825714111, 0.5215877890586853, 0.5280532836914062, 0.4484952688217163, 0.46615374088287354, 0.44937077164649963, 0.45945388078689575, 0.4651395082473755, 0.5134136080741882, 0.4537816345691681, 0.4814797639846802, 0.4958707094192505, 0.4620446264743805, 0.49143582582473755, 0.5383617877960205, 0.5094648003578186, 0.5036094188690186, 0.48586833477020264, 0.4679613709449768, 0.4366447627544403, 0.49398207664489746, 0.4735323190689087, 0.4096084535121918, 0.42639753222465515, 0.5111979842185974, 0.4262006878852844, 0.46282240748405457, 0.5198631286621094, 0.4455399215221405, 0.47774288058280945, 0.4570591449737549, 0.4492846131324768, 0.4886434078216553, 0.4240056574344635, 0.5078577995300293, 0.4600840210914612, 0.4807472229003906, 0.4507070779800415, 0.4666961431503296, 0.4485797882080078, 0.5611534714698792, 0.550327718257904, 0.542255699634552, 0.5911049246788025, 0.5134512186050415, 0.5556694269180298, 0.5282996296882629, 0.5085437893867493, 0.4948958456516266, 0.5389347672462463, 0.4833510220050812, 0.5236506462097168, 0.40729087591171265, 0.5984289050102234, 0.46314331889152527, 0.5613393187522888, 0.5228904485702515, 0.401719868183136, 0.4173804223537445]
    elif args.filename == 'FCOS-V-57-MS-FPN-2x':
        thresholds_for_classes = [0.48878026008605957, 0.5157854557037354, 0.5044764876365662, 0.49984481930732727, 0.5201855897903442, 0.5268604159355164, 0.542915403842926, 0.49880141019821167, 0.4862997531890869, 0.4703293442726135, 0.48309165239334106, 0.5980977416038513, 0.5839372873306274, 0.43605467677116394, 0.488334983587265, 0.5531968474388123, 0.6045324206352234, 0.5037994980812073, 0.5493407249450684, 0.5217279195785522, 0.5599054098129272, 0.541961133480072, 0.52456134557724, 0.590713620185852, 0.47225916385650635, 0.5106967091560364, 0.4527757167816162, 0.49706345796585083, 0.46771353483200073, 0.48096945881843567, 0.42433881759643555, 0.5834863781929016, 0.4681493937969208, 0.4610227644443512, 0.495819091796875, 0.49427446722984314, 0.5497299432754517, 0.49133870005607605, 0.5676629543304443, 0.4558263421058655, 0.4907543957233429, 0.5280252695083618, 0.4518597722053528, 0.4859154224395752, 0.4274418354034424, 0.5014893412590027, 0.44185423851013184, 0.49433377385139465, 0.5254765748977661, 0.49024903774261475, 0.47859156131744385, 0.5015891790390015, 0.4337356686592102, 0.5387176275253296, 0.45217376947402954, 0.49776363372802734, 0.46315333247184753, 0.5012166500091553, 0.46372056007385254, 0.4833259880542755, 0.4585243761539459, 0.55859375, 0.5683536529541016, 0.5178630352020264, 0.6591401100158691, 0.4779455363750458, 0.6092398762702942, 0.465324342250824, 0.5212925672531128, 0.543681800365448, 0.46491605043411255, 0.5281332731246948, 0.5284128189086914, 0.42081254720687866, 0.5965998768806458, 0.5190021991729736, 0.520578145980835, 0.5551347136497498, 0.3499980568885803, 0.5275787711143494]
    elif args.filename == 'FCOS-V-93-MS-FPN-2x':
        thresholds_for_classes = [0.5007159113883972, 0.5027709603309631, 0.4983883798122406, 0.5042306780815125, 0.5172843933105469, 0.5421777963638306, 0.49390166997909546, 0.4747004210948944, 0.45099371671676636, 0.4309789538383484, 0.5930590629577637, 0.4862874746322632, 0.5522752404212952, 0.46066027879714966, 0.4280782639980316, 0.5675986409187317, 0.5979647636413574, 0.5108857154846191, 0.5236023664474487, 0.47734686732292175, 0.5121535658836365, 0.5512094497680664, 0.5194173455238342, 0.529695451259613, 0.46202796697616577, 0.49217405915260315, 0.47130680084228516, 0.5164242386817932, 0.460493803024292, 0.51019287109375, 0.41587454080581665, 0.5646100640296936, 0.4817523956298828, 0.4282572865486145, 0.5390541553497314, 0.5457136631011963, 0.543086588382721, 0.4959389865398407, 0.5267823934555054, 0.4736594557762146, 0.48538756370544434, 0.5409947037696838, 0.4628223180770874, 0.4627744257450104, 0.46495139598846436, 0.49192482233047485, 0.43032753467559814, 0.5188134908676147, 0.5054355263710022, 0.4710555076599121, 0.49257805943489075, 0.49037864804267883, 0.47253096103668213, 0.5415661931037903, 0.4508369565010071, 0.5665602684020996, 0.4934995770454407, 0.49727723002433777, 0.46266302466392517, 0.44974660873413086, 0.46480679512023926, 0.609650194644928, 0.5211730599403381, 0.5215638279914856, 0.5687638521194458, 0.4933472275733948, 0.5420998334884644, 0.5458967685699463, 0.5249426364898682, 0.49063530564308167, 0.4493103623390198, 0.5074935555458069, 0.5498480200767517, 0.43373802304267883, 0.5894798040390015, 0.5149099826812744, 0.5526571273803711, 0.5188148617744446, 0.3811971843242645, 0.5093425512313843]

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

    tracker = Tracker(max_age=30, nn_budget=70)
    print('DeepSORT Tracker inited!')
    drawer = Drawer()

    for frame_idx in tqdm(range(total_frames)):
        start_time = time.time()
        ret_val, frame = vid_cap.read()
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bbs = coco_demo.bbs_on_opencv_image(frame_RGB, classes=['person'])  # bbs in ltwh

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
                cv2.imshow("COCO detections", show_frame)
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
