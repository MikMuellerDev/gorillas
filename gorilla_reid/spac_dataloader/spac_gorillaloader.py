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
from utils import rfile, datapoint_from_path
from datapoint import SPACDataPoint, spac_datapoint_from_dict
from triplet import GorillaTripletDataPoint


def load_datapoints(image_dir=DEFAULT_SPAC_IMAGE_DIR) -> List[SPACDataPoint]:
    """
    Loads datapoint index from disk.
    If the cache file does not exist, this can take a while because it needs
    to scan the entire file structure.
    """
    try:
        with open(DP_DB_JSON_PATH, "r", encoding="utf-8") as f:
            print("[LOAD] Using cached datapoint index...")
            loaded = json.loads(f.read())
            transformed = [spac_datapoint_from_dict(d) for d in loaded]
            return transformed
    except Exception:
        # Naive implementation, assume that any error can be ignored.
        pass

    total_files = 0

    directory = os.fsencode(image_dir)
    total_files += len(os.listdir(directory))

    datapoints = []

    pbar = tqdm(total=total_files, desc="[LOAD] Indexing...")

    for [idx, file] in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            complete_path_image = os.path.join(image_dir, filename)
            dp = datapoint_from_path(complete_path_image)
            #     gorilla_id=gorilla_id,
            #     camera=camera_id,
            #     date=date,
            #     filepath=complete_path_image,
            # )

            datapoints.append(dp)

            pbar.update(1)
            continue
        else:
            continue

    with open(DP_DB_JSON_PATH, "w", encoding="utf-8") as f:
        dp_dicts = [dp.to_dict() for dp in datapoints]
        json.dump(dp_dicts, f, indent=4, ensure_ascii=False)

    print(f"[LOAD] Saved {len(datapoints)} datapoints to '{DP_DB_JSON_PATH}'")
    return datapoints


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
            datapoints: List[SPACDataPoint],
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
                    


                
        #
        # Triplet generation.
        #
        self.dataset_indices_used_for_triplets = {}

        print(f"[DATASET]: Using data from index {start_index} to {end_index}, covering {end_index - start_index}, total {total_datapoints} DP.")

    def get_triplet(self, idx: int) -> GorillaTripletDataPoint:
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
        return GorillaTripletDataPoint(
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
