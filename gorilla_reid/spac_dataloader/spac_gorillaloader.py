import json
import random
from typing import List
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import os
from tqdm.notebook import tqdm
from IPython.display import clear_output
from . import utils
# from utils import DEFAULT_SPAC_IMAGE_DIR, DP_DB_JSON_PATH, rfile, datapoint_from_path
from . import datapoint
from . import triplet
# from datapoint import SPACDataPoint, spac_datapoint_from_dict
# from triplet import GorillaTripletDataPoint



def read_image(path, label_mapping: dict, image_dim=(224, 224)):
    """
    Exists because reading images to tensors and a label is tedious.
    Loads the image as a tensor and AUTOMATICALLY loads the correct label
    as an ID and converts it to a one-hot tensor.
    """
    filename = os.path.basename(path)

    img = Image.open(path)
    rez_tr = transforms.Resize(image_dim)
    rez_img = rez_tr(img)
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    tensor_image = transform_to_tensor(rez_img)

    label = filename.split("_")[0]
    label_id = label_mapping[label]
    tensor_one_hot = F.one_hot(
            torch.tensor(label_id),
            num_classes=len(label_mapping)
    ).float()

    return (tensor_image, tensor_one_hot)


class GorillaDataset(Dataset):
    def __init__(
            self,
            datapoints: List[datapoint.SPACDataPoint],
            percent_start: int,
            percent_end: int,
            include_transformations: bool):
        #
        # Slice.
        #
        total_datapoints = len(datapoints)
        start_index = int(total_datapoints * percent_start / 100)
        end_index = int(total_datapoints * percent_end / 100) - 1
        self.datapoints = datapoints[start_index:end_index]

        #
        # Group by gorilla-id
        #
        self.datapoints_by_gorilla_id = {}

        for dp in self.datapoints:
            id = dp.gorilla_id
            if id not in self.datapoints_by_gorilla_id:
                self.datapoints_by_gorilla_id[id] = []

            self.datapoints_by_gorilla_id[id].append(dp)

        #
        # Delete all gorillas that have less than N occurences in total.
        # We need to do this in order to get enough datapoints for triplet-loss
        #
        ensure_datapoints_per_gorilla = 2

        to_be_deleted = set()
        for gorilla_id in self.datapoints_by_gorilla_id.keys():
            if len(self.datapoints_by_gorilla_id[gorilla_id]) < ensure_datapoints_per_gorilla:
                to_be_deleted.add(gorilla_id)

        for key in to_be_deleted:
            del self.datapoints_by_gorilla_id[key]

        len_before = len(self.datapoints)
        self.datapoints = [dp for dp in self.datapoints if dp.gorilla_id in self.datapoints_by_gorilla_id]
        print(f"[DATASET] Ignoring {len_before - len(self.datapoints)} datapoints due to data distribution.")

        #
        # Data flattening: if transformations are enabled, include all transformed images in the 1-dimensional data space.
        #

        additional_flattened = []
        if include_transformations:
            for dp in self.datapoints:
                for transformed in dp.transformations:
                    additional_flattened.append(transformed)
        self.datapoints.extend(additional_flattened)

                
        #
        # Triplet generation.
        #
        self.dataset_indices_used_for_triplets = {}

        additional = ""
        if include_transformations:
            additional = f" (w. transformations: {len(self.datapoints)})"
        print(f"[DATASET]: Using data from index {start_index} to {end_index}, covering {end_index - start_index}, total {total_datapoints}.{additional}")

    def get_triplet(self, idx: int) -> triplet.GorillaTripletDataPoint:
        base_datapoint = self.datapoints[idx]
        gorilla_id = base_datapoint.gorilla_id

        # Get random element from the same class.
        samples_in_this_class = len(self.datapoints_by_gorilla_id[gorilla_id])
        if samples_in_this_class == 1:
            # This should be prevented by the ignorelist.
            raise Exception("Corrupt / malformed dataset: tried to acccess ANOTHER gorilla-dataset from the same class, but did not find any")

        in_class_index = idx

        while in_class_index == idx:
            in_class_index = random.randint(0, samples_in_this_class - 1)

        out_class_datapoint_index = -1
        out_class_gorilla_id = gorilla_id
        while out_class_gorilla_id == gorilla_id:
            out_class_key_index = random.randint(0, len(self.datapoints_by_gorilla_id) - 1)
            random_out_class_key = list(self.datapoints_by_gorilla_id.keys())[out_class_key_index]
            random_out_class = self.datapoints_by_gorilla_id[random_out_class_key]

            out_class_datapoint_index = random.randint(0, len(random_out_class) - 1)
            out_class_gorilla_id = random_out_class[out_class_datapoint_index].gorilla_id
        
        in_class_datapoint = self.datapoints_by_gorilla_id[gorilla_id][in_class_index]
        out_class_datapoint = self.datapoints_by_gorilla_id[out_class_gorilla_id][out_class_datapoint_index]
        return triplet.GorillaTripletDataPoint(
            this_class=base_datapoint,
            in_class=in_class_datapoint,
            out_class=out_class_datapoint,
        )
        
    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        normal_triplet = self.get_triplet(idx)
        triplet = normal_triplet.to_tensor()
        label_str = normal_triplet.in_class.gorilla_id
        return label_str, *triplet.to_triple()
