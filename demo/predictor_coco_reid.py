# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T

from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.structures.image_list import to_image_list
from fcos_core.modeling.roi_heads.mask_head.inference import Masker
from fcos_core import layers as L
from fcos_core.utils import cv2_util


class COCODemo(object):
	# COCO categories for pretty print
	CATEGORIES = [
		"__background",
		"person",
		"bicycle",
		"car",
		"motorcycle",
		"airplane",
		"bus",
		"train",
		"truck",
		"boat",
		"traffic light",
		"fire hydrant",
		"stop sign",
		"parking meter",
		"bench",
		"bird",
		"cat",
		"dog",
		"horse",
		"sheep",
		"cow",
		"elephant",
		"bear",
		"zebra",
		"giraffe",
		"backpack",
		"umbrella",
		"handbag",
		"tie",
		"suitcase",
		"frisbee",
		"skis",
		"snowboard",
		"sports ball",
		"kite",
		"baseball bat",
		"baseball glove",
		"skateboard",
		"surfboard",
		"tennis racket",
		"bottle",
		"wine glass",
		"cup",
		"fork",
		"knife",
		"spoon",
		"bowl",
		"banana",
		"apple",
		"sandwich",
		"orange",
		"broccoli",
		"carrot",
		"hot dog",
		"pizza",
		"donut",
		"cake",
		"chair",
		"couch",
		"potted plant",
		"bed",
		"dining table",
		"toilet",
		"tv",
		"laptop",
		"mouse",
		"remote",
		"keyboard",
		"cell phone",
		"microwave",
		"oven",
		"toaster",
		"sink",
		"refrigerator",
		"book",
		"clock",
		"vase",
		"scissors",
		"teddy bear",
		"hair drier",
		"toothbrush",
	]

	def __init__(
		self,
		cfg,
		filename='FCOS-V-57-MS-FPN-2x',
		show_mask_heatmaps=False,
		masks_per_dim=2,
		min_image_size=800,
	):
		self.cfg = cfg.clone()
		self.model = build_detection_model(cfg)
		self.model.eval()
		self.device = torch.device(cfg.MODEL.DEVICE)
		self.model.to(self.device)
		self.min_image_size = min_image_size

		save_dir = cfg.OUTPUT_DIR
		checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
		_ = checkpointer.load(cfg.MODEL.WEIGHT)

		self.transforms = self.build_transform()

		mask_threshold = -1 if show_mask_heatmaps else 0.5
		self.masker = Masker(threshold=mask_threshold, padding=1)

		# used to make colors for each class
		self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

		self.cpu_device = torch.device("cpu")
		self.show_mask_heatmaps = show_mask_heatmaps
		self.masks_per_dim = masks_per_dim

		if filename == 'FCOS-V-39-FPN-1x':
			confidence_thresholds_for_classes = [0.4871613681316376, 0.45935359597206116, 0.47524023056030273, 0.46535882353782654, 0.5457630753517151, 0.5112412571907043, 0.5289180278778076, 0.4660639464855194, 0.45085862278938293, 0.47864195704460144, 0.4975494146347046, 0.4788793921470642, 0.5068866014480591, 0.4542357325553894, 0.4678514003753662, 0.4941931366920471, 0.5626294612884521, 0.5301299095153809, 0.5370022058486938, 0.5048402547836304, 0.4960610270500183, 0.6174976825714111, 0.5215877890586853, 0.5280532836914062, 0.4484952688217163, 0.46615374088287354, 0.44937077164649963, 0.45945388078689575, 0.4651395082473755, 0.5134136080741882, 0.4537816345691681, 0.4814797639846802, 0.4958707094192505, 0.4620446264743805, 0.49143582582473755, 0.5383617877960205, 0.5094648003578186, 0.5036094188690186, 0.48586833477020264, 0.4679613709449768, 0.4366447627544403, 0.49398207664489746, 0.4735323190689087, 0.4096084535121918, 0.42639753222465515, 0.5111979842185974, 0.4262006878852844, 0.46282240748405457, 0.5198631286621094, 0.4455399215221405, 0.47774288058280945, 0.4570591449737549, 0.4492846131324768, 0.4886434078216553, 0.4240056574344635, 0.5078577995300293, 0.4600840210914612, 0.4807472229003906, 0.4507070779800415, 0.4666961431503296, 0.4485797882080078, 0.5611534714698792, 0.550327718257904, 0.542255699634552, 0.5911049246788025, 0.5134512186050415, 0.5556694269180298, 0.5282996296882629, 0.5085437893867493, 0.4948958456516266, 0.5389347672462463, 0.4833510220050812, 0.5236506462097168, 0.40729087591171265, 0.5984289050102234, 0.46314331889152527, 0.5613393187522888, 0.5228904485702515, 0.401719868183136, 0.4173804223537445]
		elif filename == 'FCOS-V-57-MS-FPN-2x':
			confidence_thresholds_for_classes = [0.48878026008605957, 0.5157854557037354, 0.5044764876365662, 0.49984481930732727, 0.5201855897903442, 0.5268604159355164, 0.542915403842926, 0.49880141019821167, 0.4862997531890869, 0.4703293442726135, 0.48309165239334106, 0.5980977416038513, 0.5839372873306274, 0.43605467677116394, 0.488334983587265, 0.5531968474388123, 0.6045324206352234, 0.5037994980812073, 0.5493407249450684, 0.5217279195785522, 0.5599054098129272, 0.541961133480072, 0.52456134557724, 0.590713620185852, 0.47225916385650635, 0.5106967091560364, 0.4527757167816162, 0.49706345796585083, 0.46771353483200073, 0.48096945881843567, 0.42433881759643555, 0.5834863781929016, 0.4681493937969208, 0.4610227644443512, 0.495819091796875, 0.49427446722984314, 0.5497299432754517, 0.49133870005607605, 0.5676629543304443, 0.4558263421058655, 0.4907543957233429, 0.5280252695083618, 0.4518597722053528, 0.4859154224395752, 0.4274418354034424, 0.5014893412590027, 0.44185423851013184, 0.49433377385139465, 0.5254765748977661, 0.49024903774261475, 0.47859156131744385, 0.5015891790390015, 0.4337356686592102, 0.5387176275253296, 0.45217376947402954, 0.49776363372802734, 0.46315333247184753, 0.5012166500091553, 0.46372056007385254, 0.4833259880542755, 0.4585243761539459, 0.55859375, 0.5683536529541016, 0.5178630352020264, 0.6591401100158691, 0.4779455363750458, 0.6092398762702942, 0.465324342250824, 0.5212925672531128, 0.543681800365448, 0.46491605043411255, 0.5281332731246948, 0.5284128189086914, 0.42081254720687866, 0.5965998768806458, 0.5190021991729736, 0.520578145980835, 0.5551347136497498, 0.3499980568885803, 0.5275787711143494]
		elif filename == 'FCOS-V-93-MS-FPN-2x':
			confidence_thresholds_for_classes = [0.5007159113883972, 0.5027709603309631, 0.4983883798122406, 0.5042306780815125, 0.5172843933105469, 0.5421777963638306, 0.49390166997909546, 0.4747004210948944, 0.45099371671676636, 0.4309789538383484, 0.5930590629577637, 0.4862874746322632, 0.5522752404212952, 0.46066027879714966, 0.4280782639980316, 0.5675986409187317, 0.5979647636413574, 0.5108857154846191, 0.5236023664474487, 0.47734686732292175, 0.5121535658836365, 0.5512094497680664, 0.5194173455238342, 0.529695451259613, 0.46202796697616577, 0.49217405915260315, 0.47130680084228516, 0.5164242386817932, 0.460493803024292, 0.51019287109375, 0.41587454080581665, 0.5646100640296936, 0.4817523956298828, 0.4282572865486145, 0.5390541553497314, 0.5457136631011963, 0.543086588382721, 0.4959389865398407, 0.5267823934555054, 0.4736594557762146, 0.48538756370544434, 0.5409947037696838, 0.4628223180770874, 0.4627744257450104, 0.46495139598846436, 0.49192482233047485, 0.43032753467559814, 0.5188134908676147, 0.5054355263710022, 0.4710555076599121, 0.49257805943489075, 0.49037864804267883, 0.47253096103668213, 0.5415661931037903, 0.4508369565010071, 0.5665602684020996, 0.4934995770454407, 0.49727723002433777, 0.46266302466392517, 0.44974660873413086, 0.46480679512023926, 0.609650194644928, 0.5211730599403381, 0.5215638279914856, 0.5687638521194458, 0.4933472275733948, 0.5420998334884644, 0.5458967685699463, 0.5249426364898682, 0.49063530564308167, 0.4493103623390198, 0.5074935555458069, 0.5498480200767517, 0.43373802304267883, 0.5894798040390015, 0.5149099826812744, 0.5526571273803711, 0.5188148617744446, 0.3811971843242645, 0.5093425512313843]
		self.confidence_thresholds_for_classes = torch.tensor(confidence_thresholds_for_classes)

	def build_transform(self):
		"""
		Creates a basic transformation that was used to train the models
		"""
		cfg = self.cfg

		# we are loading images with OpenCV, so we don't need to convert them
		# to BGR, they are already! So all we need to do is to normalize
		# by 255 if we want to convert to BGR255 format, or flip the channels
		# if we want it to be in RGB in [0-1] range.
		if cfg.INPUT.TO_BGR255:
			to_bgr_transform = T.Lambda(lambda x: x * 255)
		else:
			to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

		normalize_transform = T.Normalize(
			mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
		)

		transform = T.Compose(
			[
				T.ToPILImage(),
				T.Resize(self.min_image_size),
				T.ToTensor(),
				to_bgr_transform,
				normalize_transform,
			]
		)
		return transform

	def run_on_opencv_image(self, image):
		"""
		Arguments:
			image (np.ndarray): an image as returned by OpenCV

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		predictions, _ = self.compute_prediction(image)
		top_predictions = self.select_top_predictions(predictions)

		result = image.copy()
		if self.show_mask_heatmaps:
			return self.create_mask_montage(result, top_predictions)
		result = self.overlay_boxes(result, top_predictions)
		if self.cfg.MODEL.MASK_ON:
			result = self.overlay_mask(result, top_predictions)
		if self.cfg.MODEL.KEYPOINT_ON:
			result = self.overlay_keypoints(result, top_predictions)
		result = self.overlay_class_names(result, top_predictions)

		return result

	def bbs_on_opencv_image(self, image, classes=None):
		"""
		Arguments:
			image (np.ndarray): an image as returned by OpenCV

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		predictions = self.compute_prediction(image)
		top_predictions = self.select_top_predictions(predictions)

		result = self.get_boxes(top_predictions, classes)

		return result

	def compute_prediction(self, original_image):
		"""
		Arguments:
			original_image (np.ndarray): an image as returned by OpenCV

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		# apply pre-processing to image
		image = self.transforms(original_image)
		# convert to an ImageList, padded so that it is divisible by
		# cfg.DATALOADER.SIZE_DIVISIBILITY
		image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
		image_list = image_list.to(self.device)
		# compute predictions
		with torch.no_grad():
			predictions = self.model(image_list)
		predictions = [o.to(self.cpu_device) for o in predictions]

		# always single image is passed at a time
		prediction = predictions[0]

		# reshape prediction (a BoxList) into the original image size
		height, width = original_image.shape[:-1]
		prediction = prediction.resize((width, height))

		if prediction.has_field("mask"):
			# if we have masks, paste the masks in the right position
			# in the image, as defined by the bounding boxes
			masks = prediction.get_field("mask")
			# always single image is passed at a time
			masks = self.masker([masks], [prediction])[0]
			prediction.add_field("mask", masks)
		return prediction

	def select_top_predictions(self, predictions):
		"""
		Select only predictions which have a `score` > self.confidence_threshold,
		and returns the predictions in descending order of score

		Arguments:
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `scores`.

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		scores = predictions.get_field("scores")
		labels = predictions.get_field("labels")
		thresholds = self.confidence_thresholds_for_classes[(labels - 1).long()]
		keep = torch.nonzero(scores > thresholds).squeeze(1)
		predictions = predictions[keep]
		scores = predictions.get_field("scores")
		_, idx = scores.sort(0, descending=True)
		return predictions[idx]

	def compute_colors_for_labels(self, labels):
		"""
		Simple function that adds fixed colors depending on the class
		"""
		colors = labels[:, None] * self.palette
		colors = (colors % 255).numpy().astype("uint8")
		return colors

	def overlay_boxes(self, image, predictions):
		"""
		Adds the predicted boxes on top of the image

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `labels`.
		"""
		labels = predictions.get_field("labels")
		boxes = predictions.bbox

		colors = self.compute_colors_for_labels(labels).tolist()

		for box, color in zip(boxes, colors):
			box = box.to(torch.int64)
			top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
			image = cv2.rectangle(
				image, tuple(top_left), tuple(bottom_right), tuple(color), 2
			)

		return image

	def get_boxes(self, predictions, classes=None):
		"""
		Adds the predicted boxes on top of the image

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `labels`.
		"""
		scores = predictions.get_field("scores").tolist()
		labels = predictions.get_field("labels").tolist()
		labels = [self.CATEGORIES[i] for i in labels]
		boxes = predictions.bbox # mode: xyxy

		bbs = []
		for box, score, label in zip(boxes, scores, labels):
			if classes is None or label in classes:
				box_infos = []
				box = box.to(torch.int64)

				x1, y1, x2, y2 = box.tolist()
				# append as ltwh
				box_infos.append(int(x1))
				box_infos.append(int(y1))
				box_infos.append(int(x2-x1))
				box_infos.append(int(y2-y1))
				assert len(box_infos) > 0 ,'box infos is blank'

				bbs.append((box_infos, score, label))

		return bbs

	def overlay_mask(self, image, predictions):
		"""
		Adds the instances contours for each predicted object.
		Each label has a different color.

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `mask` and `labels`.
		"""
		masks = predictions.get_field("mask").numpy()
		labels = predictions.get_field("labels")

		colors = self.compute_colors_for_labels(labels).tolist()

		for mask, color in zip(masks, colors):
			thresh = mask[0, :, :, None]
			contours, hierarchy = cv2_util.findContours(
				thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
			)
			image = cv2.drawContours(image, contours, -1, color, 3)

		composite = image

		return composite

	def overlay_keypoints(self, image, predictions):
		keypoints = predictions.get_field("keypoints")
		kps = keypoints.keypoints
		scores = keypoints.get_field("logits")
		kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
		for region in kps:
			image = vis_keypoints(image, region.transpose((1, 0)))
		return image

	def create_mask_montage(self, image, predictions):
		"""
		Create a montage showing the probability heatmaps for each one one of the
		detected objects

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `mask`.
		"""
		masks = predictions.get_field("mask")
		masks_per_dim = self.masks_per_dim
		masks = L.interpolate(
			masks.float(), scale_factor=1 / masks_per_dim
		).byte()
		height, width = masks.shape[-2:]
		max_masks = masks_per_dim ** 2
		masks = masks[:max_masks]
		# handle case where we have less detections than max_masks
		if len(masks) < max_masks:
			masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
			masks_padded[: len(masks)] = masks
			masks = masks_padded
		masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
		result = torch.zeros(
			(masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
		)
		for y in range(masks_per_dim):
			start_y = y * height
			end_y = (y + 1) * height
			for x in range(masks_per_dim):
				start_x = x * width
				end_x = (x + 1) * width
				result[start_y:end_y, start_x:end_x] = masks[y, x]
		return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

	def overlay_class_names(self, image, predictions):
		"""
		Adds detected class names and scores in the positions defined by the
		top-left corner of the predicted bounding box

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `scores` and `labels`.
		"""
		scores = predictions.get_field("scores").tolist()
		labels = predictions.get_field("labels").tolist()
		labels = [self.CATEGORIES[i] for i in labels]
		boxes = predictions.bbox

		template = "{}: {:.2f}"
		for box, score, label in zip(boxes, scores, labels):
			x, y = box[:2]
			s = template.format(label, score)
			cv2.putText(
				image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
			)

		return image

	def get_boxes_reid(self, predictions, classes=None):
		"""
		Adds the predicted boxes on top of the image

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `labels`.
		"""
		scores = predictions.get_field("scores").tolist()
		labels = predictions.get_field("labels").tolist()
		labels = [self.CATEGORIES[i] for i in labels]
		boxes = predictions.bbox # mode: xyxy

		bbs = []
		for box, score, label in zip(boxes, scores, labels):
			if classes is None or label in classes:
				box = box.to(torch.int64)

				x1, y1, x2, y2 = box.tolist()
				left = int(x1)
				top = int(y1)
				bot = int(y2)
				right = int(x2)
				width = int(x2-x1)
				height = int(y2-y1)

				bbs.append( {'label':label,'confidence':score,'t':top,'l':left,'b':bot,'r':right,'w':width,'h':height} )

		return bbs

	def get_detections_dict(self, frames, classes=None):
		'''
		Params: frames, list of ndarray-like
		Returns: detections, list of dict, whose key: label, confidence, t, l, w, h
		'''
		predictions = self.compute_prediction(frames[0])
		top_predictions = self.select_top_predictions(predictions)

		all_detections = self.get_boxes_reid(top_predictions, classes)
		
		return [all_detections]

import numpy as np
import matplotlib.pyplot as plt
from fcos_core.structures.keypoint import PersonKeypoints

def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
	"""Visualizes keypoints (adapted from vis_one_image).
	kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
	"""
	dataset_keypoints = PersonKeypoints.NAMES
	kp_lines = PersonKeypoints.CONNECTIONS

	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
	colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

	# Perform the drawing on a copy of the image, to allow for blending.
	kp_mask = np.copy(img)

	# Draw mid shoulder / mid hip first for better visualization.
	mid_shoulder = (
		kps[:2, dataset_keypoints.index('right_shoulder')] +
		kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
	sc_mid_shoulder = np.minimum(
		kps[2, dataset_keypoints.index('right_shoulder')],
		kps[2, dataset_keypoints.index('left_shoulder')])
	mid_hip = (
		kps[:2, dataset_keypoints.index('right_hip')] +
		kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
	sc_mid_hip = np.minimum(
		kps[2, dataset_keypoints.index('right_hip')],
		kps[2, dataset_keypoints.index('left_hip')])
	nose_idx = dataset_keypoints.index('nose')
	if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
		cv2.line(
			kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
			color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
	if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
		cv2.line(
			kp_mask, tuple(mid_shoulder), tuple(mid_hip),
			color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

	# Draw the keypoints.
	for l in range(len(kp_lines)):
		i1 = kp_lines[l][0]
		i2 = kp_lines[l][1]
		p1 = kps[0, i1], kps[1, i1]
		p2 = kps[0, i2], kps[1, i2]
		if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
			cv2.line(
				kp_mask, p1, p2,
				color=colors[l], thickness=2, lineType=cv2.LINE_AA)
		if kps[2, i1] > kp_thresh:
			cv2.circle(
				kp_mask, p1,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
		if kps[2, i2] > kp_thresh:
			cv2.circle(
				kp_mask, p2,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

	# Blend the keypoints.
	return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
