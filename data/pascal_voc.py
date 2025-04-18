import pickle
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from utils.config import cfg
from utils.utils import lexico_iter

anno_path = cfg.VOC2011.KPT_ANNO_DIR
img_path = cfg.VOC2011.ROOT_DIR + "JPEGImages"
ori_anno_path = cfg.VOC2011.ROOT_DIR + "Annotations"
set_path = cfg.VOC2011.SET_SPLIT
cache_path = cfg.CACHE_PATH

KPT_NAMES = {
    "cat": [
        "L_B_Elbow",
        "L_B_Paw",
        "L_EarBase",
        "L_Eye",
        "L_F_Elbow",
        "L_F_Paw",
        "Nose",
        "R_B_Elbow",
        "R_B_Paw",
        "R_EarBase",
        "R_Eye",
        "R_F_Elbow",
        "R_F_Paw",
        "TailBase",
        "Throat",
        "Withers",
    ],
    "bottle": ["L_Base", "L_Neck", "L_Shoulder", "L_Top", "R_Base", "R_Neck", "R_Shoulder", "R_Top"],
    "horse": [
        "L_B_Elbow",
        "L_B_Paw",
        "L_EarBase",
        "L_Eye",
        "L_F_Elbow",
        "L_F_Paw",
        "Nose",
        "R_B_Elbow",
        "R_B_Paw",
        "R_EarBase",
        "R_Eye",
        "R_F_Elbow",
        "R_F_Paw",
        "TailBase",
        "Throat",
        "Withers",
    ],
    "motorbike": [
        "B_WheelCenter",
        "B_WheelEnd",
        "ExhaustPipeEnd",
        "F_WheelCenter",
        "F_WheelEnd",
        "HandleCenter",
        "L_HandleTip",
        "R_HandleTip",
        "SeatBase",
        "TailLight",
    ],
    "boat": [
        "Hull_Back_Bot",
        "Hull_Back_Top",
        "Hull_Front_Bot",
        "Hull_Front_Top",
        "Hull_Mid_Left_Bot",
        "Hull_Mid_Left_Top",
        "Hull_Mid_Right_Bot",
        "Hull_Mid_Right_Top",
        "Mast_Top",
        "Sail_Left",
        "Sail_Right",
    ],
    "tvmonitor": [
        "B_Bottom_Left",
        "B_Bottom_Right",
        "B_Top_Left",
        "B_Top_Right",
        "F_Bottom_Left",
        "F_Bottom_Right",
        "F_Top_Left",
        "F_Top_Right",
    ],
    "cow": [
        "L_B_Elbow",
        "L_B_Paw",
        "L_EarBase",
        "L_Eye",
        "L_F_Elbow",
        "L_F_Paw",
        "Nose",
        "R_B_Elbow",
        "R_B_Paw",
        "R_EarBase",
        "R_Eye",
        "R_F_Elbow",
        "R_F_Paw",
        "TailBase",
        "Throat",
        "Withers",
    ],
    "chair": [
        "BackRest_Top_Left",
        "BackRest_Top_Right",
        "Leg_Left_Back",
        "Leg_Left_Front",
        "Leg_Right_Back",
        "Leg_Right_Front",
        "Seat_Left_Back",
        "Seat_Left_Front",
        "Seat_Right_Back",
        "Seat_Right_Front",
    ],
    "car": [
        "L_B_RoofTop",
        "L_B_WheelCenter",
        "L_F_RoofTop",
        "L_F_WheelCenter",
        "L_HeadLight",
        "L_SideviewMirror",
        "L_TailLight",
        "R_B_RoofTop",
        "R_B_WheelCenter",
        "R_F_RoofTop",
        "R_F_WheelCenter",
        "R_HeadLight",
        "R_SideviewMirror",
        "R_TailLight",
    ],
    "person": [
        "B_Head",
        "HeadBack",
        "L_Ankle",
        "L_Ear",
        "L_Elbow",
        "L_Eye",
        "L_Foot",
        "L_Hip",
        "L_Knee",
        "L_Shoulder",
        "L_Toes",
        "L_Wrist",
        "Nose",
        "R_Ankle",
        "R_Ear",
        "R_Elbow",
        "R_Eye",
        "R_Foot",
        "R_Hip",
        "R_Knee",
        "R_Shoulder",
        "R_Toes",
        "R_Wrist",
    ],
    "diningtable": [
        "Bot_Left_Back",
        "Bot_Left_Front",
        "Bot_Right_Back",
        "Bot_Right_Front",
        "Top_Left_Back",
        "Top_Left_Front",
        "Top_Right_Back",
        "Top_Right_Front",
    ],
    "dog": [
        "L_B_Elbow",
        "L_B_Paw",
        "L_EarBase",
        "L_Eye",
        "L_F_Elbow",
        "L_F_Paw",
        "Nose",
        "R_B_Elbow",
        "R_B_Paw",
        "R_EarBase",
        "R_Eye",
        "R_F_Elbow",
        "R_F_Paw",
        "TailBase",
        "Throat",
        "Withers",
    ],
    "bird": [
        "Beak_Base",
        "Beak_Tip",
        "Left_Eye",
        "Left_Wing_Base",
        "Left_Wing_Tip",
        "Leg_Center",
        "Lower_Neck_Base",
        "Right_Eye",
        "Right_Wing_Base",
        "Right_Wing_Tip",
        "Tail_Tip",
        "Upper_Neck_Base",
    ],
    "bicycle": [
        "B_WheelCenter",
        "B_WheelEnd",
        "B_WheelIntersection",
        "CranksetCenter",
        "F_WheelCenter",
        "F_WheelEnd",
        "F_WheelIntersection",
        "HandleCenter",
        "L_HandleTip",
        "R_HandleTip",
        "SeatBase",
    ],
    "train": [
        "Base_Back_Left",
        "Base_Back_Right",
        "Base_Front_Left",
        "Base_Front_Right",
        "Roof_Back_Left",
        "Roof_Back_Right",
        "Roof_Front_Middle",
    ],
    "sheep": [
        "L_B_Elbow",
        "L_B_Paw",
        "L_EarBase",
        "L_Eye",
        "L_F_Elbow",
        "L_F_Paw",
        "Nose",
        "R_B_Elbow",
        "R_B_Paw",
        "R_EarBase",
        "R_Eye",
        "R_F_Elbow",
        "R_F_Paw",
        "TailBase",
        "Throat",
        "Withers",
    ],
    "aeroplane": [
        "Bot_Rudder",
        "Bot_Rudder_Front",
        "L_Stabilizer",
        "L_WingTip",
        "Left_Engine_Back",
        "Left_Engine_Front",
        "Left_Wing_Base",
        "NoseTip",
        "Nose_Bottom",
        "Nose_Top",
        "R_Stabilizer",
        "R_WingTip",
        "Right_Engine_Back",
        "Right_Engine_Front",
        "Right_Wing_Base",
        "Top_Rudder",
    ],
    "sofa": [
        "Back_Base_Left",
        "Back_Base_Right",
        "Back_Top_Left",
        "Back_Top_Right",
        "Front_Base_Left",
        "Front_Base_Right",
        "Handle_Front_Left",
        "Handle_Front_Right",
        "Handle_Left_Junction",
        "Handle_Right_Junction",
        "Left_Junction",
        "Right_Junction",
    ],
    "pottedplant": ["Bottom_Left", "Bottom_Right", "Top_Back_Middle", "Top_Front_Middle", "Top_Left", "Top_Right"],
    "bus": ["L_B_Base", "L_B_RoofTop", "L_F_Base", "L_F_RoofTop", "R_B_Base", "R_B_RoofTop", "R_F_Base", "R_F_RoofTop"],
}


class PascalVOC:
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        self.classes = cfg.VOC2011.CLASSES
        self.kpt_len = [len(KPT_NAMES[_]) for _ in cfg.VOC2011.CLASSES]

        self.classes_kpts = {cls: len(KPT_NAMES[cls]) for cls in self.classes}

        self.anno_path = Path(anno_path)
        self.img_path = Path(img_path)
        self.ori_anno_path = Path(ori_anno_path)
        self.obj_resize = obj_resize
        self.sets = sets

        assert sets in ["train", "test"], "No match found for dataset {}".format(sets)
        cache_name = "voc_db_" + sets + ".pkl"
        self.cache_path = Path(cache_path)
        self.cache_file = self.cache_path / cache_name
        if self.cache_file.exists():
            with self.cache_file.open(mode="rb") as f:
                self.xml_list = pickle.load(f)
            print("xml list loaded from {}".format(self.cache_file))
        else:
            print("Caching xml list to {}...".format(self.cache_file))
            self.cache_path.mkdir(exist_ok=True, parents=True)
            with np.load(set_path, allow_pickle=True) as f:
                self.xml_list = f[sets]
            before_filter = sum([len(k) for k in self.xml_list])
            self.filter_list()
            after_filter = sum([len(k) for k in self.xml_list])
            with self.cache_file.open(mode="wb") as f:
                pickle.dump(self.xml_list, f)
            print("Filtered {} images to {}. Annotation saved.".format(before_filter, after_filter))

    def filter_list(self):
        """
        Filter out 'truncated', 'occluded' and 'difficult' images following the practice of previous works.
        In addition, this dataset has uncleaned label (in person category). They are omitted as suggested by README.
        """
        for cls_id in range(len(self.classes)):
            to_del = []
            for xml_name in self.xml_list[cls_id]:
                xml_comps = xml_name.split("/")[-1].strip(".xml").split("_")
                ori_xml_name = "_".join(xml_comps[:-1]) + ".xml"
                voc_idx = int(xml_comps[-1])
                xml_file = self.ori_anno_path / ori_xml_name
                assert xml_file.exists(), "{} does not exist.".format(xml_file)
                tree = ET.parse(xml_file.open())
                root = tree.getroot()
                obj = root.findall("object")[voc_idx - 1]

                difficult = obj.find("difficult")
                if difficult is not None:
                    difficult = int(difficult.text)
                occluded = obj.find("occluded")
                if occluded is not None:
                    occluded = int(occluded.text)
                truncated = obj.find("truncated")
                if truncated is not None:
                    truncated = int(truncated.text)
                if difficult or occluded or truncated:
                    to_del.append(xml_name)
                    continue

                # Exclude uncleaned images
                if self.classes[cls_id] == "person" and int(xml_comps[0]) > 2008:
                    to_del.append(xml_name)
                    continue

                # Exclude overlapping images in Willow
                if cfg.exclude_willow_classes:
                    if (
                            self.sets == "train"
                            and (self.classes[cls_id] == "motorbike" or self.classes[cls_id] == "car")
                            and int(xml_comps[0]) == 2007
                    ):
                        to_del.append(xml_name)
                        continue

            for x in to_del:
                self.xml_list[cls_id].remove(x)

    def get_k_samples(self, idx, k, mode, cls=None, shuffle=True, num_iterations=200):
        """
        Randomly get a sample of k objects from VOC-Berkeley keypoints dataset
        :param idx: Index of datapoint to sample, None for random sampling
        :param k: number of datapoints in sample
        :param mode: sampling strategy
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :param num_iterations: maximum number of iterations for sampling a datapoint
        :return: (k samples of data, k \choose 2 groundtruth permutation matrices)
        """
        if idx is not None:
            raise NotImplementedError("No indexed sampling implemented for PVOC.")
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        if mode == "superset" and k == 2:  # superset sampling only valid for pairs
            anno_list, perm_mat = self.get_pair_superset(cls=cls, shuffle=shuffle, num_iterations=num_iterations)
            return anno_list, [perm_mat]
        elif mode == "intersection":
            for i in range(num_iterations):
                xml_used = list(random.sample(self.xml_list[cls], 2))
                anno_dict_1, anno_dict_2 = [self.__get_anno_dict(xml, cls) for xml in xml_used]
                kp_names_1 = [keypoint["name"] for keypoint in anno_dict_1["keypoints"]]
                kp_names_2 = [keypoint["name"] for keypoint in anno_dict_2["keypoints"]]
                kp_names_filtered = set(kp_names_1).intersection(kp_names_2)
                anno_dict_1["keypoints"] = [kp for kp in anno_dict_1["keypoints"] if kp["name"] in kp_names_2]
                anno_dict_2["keypoints"] = [kp for kp in anno_dict_2["keypoints"] if kp["name"] in kp_names_1]
                anno_list = [anno_dict_1, anno_dict_2]
                
                for j in range(num_iterations):
                    if j > 2 * len(self.xml_list[cls]) or len(anno_list) == k:
                        break
                    xml = random.choice(self.xml_list[cls])
                    anno_dict = self.__get_anno_dict(xml, cls)
                    anno_dict["keypoints"] = [kp for kp in anno_dict["keypoints"] if kp["name"] in kp_names_filtered]
                    if len(anno_dict["keypoints"]) > len(kp_names_filtered) // 2 and xml not in xml_used:
                        xml_used.append(xml)
                        anno_list.append(anno_dict)
                if len(anno_list) == k:  # k samples found that match restrictions
                    break
            assert len(anno_list) == k
        elif mode == "all":
            anno_list = []
            for xml_name in random.sample(self.xml_list[cls], k):
                anno_dict = self.__get_anno_dict(xml_name, cls)
                if shuffle:
                    random.shuffle(anno_dict["keypoints"])
                anno_list.append(anno_dict)

        if shuffle:
            for anno_dict in anno_list:
                random.shuffle(anno_dict["keypoints"])

        # build permutation matrices
        perm_mat_list = [
            np.zeros([len(_["keypoints"]) for _ in anno_pair], dtype=np.float32) for anno_pair in lexico_iter(anno_list)
        ]
        for n, (s1, s2) in enumerate(lexico_iter(anno_list)):
            for i, keypoint in enumerate(s1["keypoints"]):
                for j, _keypoint in enumerate(s2["keypoints"]):
                    if keypoint["name"] == _keypoint["name"]:
                        perm_mat_list[n][i, j] = 1
        # print("----------------------------------------------------")
        # print(anno_list)
        # print("----------------------------------------------------")
        # print(perm_mat_list)
        # br
        return anno_list, perm_mat_list

    def get_pair_superset(self, cls=None, shuffle=True, num_iterations=200):
        """
        Randomly get a pair of objects from VOC-Berkeley keypoints dataset using superset sampling
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = None

        anno_dict_1 = self.__get_anno_dict(random.sample(self.xml_list[cls], 1)[0], cls)
        if shuffle:
            random.shuffle(anno_dict_1["keypoints"])
        keypoints_1 = set([kp["name"] for kp in anno_dict_1["keypoints"]])

        for xml_name in random.sample(self.xml_list[cls], min(len(self.xml_list[cls]), num_iterations)):
            anno_dict_2 = self.__get_anno_dict(xml_name, cls)
            if shuffle:
                random.shuffle(anno_dict_2["keypoints"])
            keypoints_2 = set([kp["name"] for kp in anno_dict_2["keypoints"]])
            if keypoints_1.issubset(keypoints_2):
                anno_pair = [anno_dict_1, anno_dict_2]
                break

        if anno_pair is None:
            return self.get_pair_superset(cls, shuffle, num_iterations)

        perm_mat = np.zeros([len(_["keypoints"]) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]["keypoints"]):
            for j, _keypoint in enumerate(anno_pair[1]["keypoints"]):
                if keypoint["name"] == _keypoint["name"]:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break

        assert len(row_list) == len(anno_pair[0]["keypoints"])

        return anno_pair, perm_mat

    def __get_anno_dict(self, xml_name, cls):
        """
        Get an annotation dict from xml file
        """
        xml_file = self.anno_path / xml_name
        assert xml_file.exists(), "{} does not exist.".format(xml_file)

        tree = ET.parse(xml_file.open())
        root = tree.getroot()

        img_name = root.find("./image").text + ".jpg"
        img_file = self.img_path / img_name
        bounds = root.find("./visible_bounds").attrib

        h = float(bounds["height"])
        w = float(bounds["width"])
        xmin = float(bounds["xmin"])
        ymin = float(bounds["ymin"])

        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))

        keypoint_list = []
        for keypoint in root.findall("./keypoints/keypoint"):
            attr = keypoint.attrib
            attr["x"] = (float(attr["x"]) - xmin) * self.obj_resize[0] / w
            attr["y"] = (float(attr["y"]) - ymin) * self.obj_resize[1] / h
            if -1e-5 < attr["x"] < self.obj_resize[0] + 1e-5 and -1e-5 < attr["y"] < self.obj_resize[1] + 1e-5:
                keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict["image"] = obj
        anno_dict["keypoints"] = keypoint_list
        anno_dict["bounds"] = xmin, ymin, w, h
        anno_dict["ori_sizes"] = ori_sizes
        anno_dict["cls"] = self.classes[cls]

        return anno_dict
