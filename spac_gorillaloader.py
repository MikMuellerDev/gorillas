import json
import random
from typing import List
import torch
from torchvision.io import decode_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import os
from tqdm.notebook import tqdm
from json import JSONEncoder
import time
from IPython.display import clear_output


SPAC_SYNTH_PATH = "./SPAC_SYNTHETIC"


class DataPoint():
    gorilla_id: str
    camera: str
    date: str
    # //////
    filepath: str
    transformations: List[str]

    def __init__(self, gorilla_id: str, camera: str, date: str, filepath: str, transformations: List[str]):
        self.gorilla_id = gorilla_id
        self.camera = camera
        self.date = date
        self.filepath = filepath
        self.transformations = transformations

    def to_dict(self):
        return self.__dict__


def dp_from_dict(dictionary) -> DataPoint:
    _self = DataPoint(
        gorilla_id="",
        camera="",
        date="",
        filepath="",
        transformations=[],
    )
    for k, v in dictionary.items():
        setattr(_self, k, v) 
    return _self


def get_modified_filelist(base: str) -> List[str]:
    out = []

    spac = SPAC_SYNTH_PATH
    directory = os.fsencode(spac)
    for [idx, file] in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        
        original_name = filename.split("_rotated")[0]

        if original_name == base:
            out.append(filename)

    clear_output(wait=False)
    print(f"Found {len(out)} matching transformations for {base}")
    return out


DP_DB_JSON_PATH = "./json/dp_db.json"

def load_datapoints() -> List[DataPoint]: 
    dir = "/scratch2/gorillawatch/data/SPAC_face_images/face_images/"


    try:
        with open(DP_DB_JSON_PATH, "r", encoding="utf-8") as f:
            print("Using DP cache...")
            loaded = json.loads(f.read())
            transformed = [ dp_from_dict(d) for d in loaded ]
            return transformed
    except Exception as e:
        print(f"EX: {e}")
        pass

    
    datapoints = []

    total_files = 0
    directory = os.fsencode(dir)
    total_files += len(os.listdir(directory))

    pbar = tqdm(total=total_files, desc="(Datapoint Cache) Indexing dataset...")

    directory = os.fsencode(dir)

    for [idx, file] in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            complete_path_image = os.path.join(dir, filename)

            path_segments = filename.split("_")
            gorilla_id = path_segments[0]
            camera_id = path_segments[1]
            date = path_segments[2]
            
            dp = DataPoint(
                gorilla_id=gorilla_id,
                camera=camera_id,
                date=date,
                filepath=complete_path_image,
                transformations=get_modified_filelist(filename)
            )

            datapoints.append(dp)

            pbar.update(1)
            continue
        else:
            continue
    
    
    with open(DP_DB_JSON_PATH, "w", encoding="utf-8") as f:
        dp_dicts = [dp.to_dict() for dp in datapoints]
        json.dump(dp_dicts, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(datapoints)} datapoints to '{DP_DB_JSON_PATH}'")
    return datapoints


def read_image(path, label_mapping: dict):
    filename = os.path.basename(path)

    img = Image.open(path)
    rez_tr = transforms.Resize((224, 224))
    rez_img = rez_tr(img)
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    tensor_image = transform_to_tensor(rez_img)

    label = filename.split("_")[0]
    label_id = label_mapping[label]
    tensor_one_hot = F.one_hot(torch.tensor(label_id), num_classes=len(label_mapping)).float()

    return (tensor_image, tensor_one_hot)

    
class GorillaTripletTensor():
    this_class: torch.Tensor
    in_class: torch.Tensor
    out_class: torch.Tensor

    def __init__(self, this_class: torch.Tensor, in_class: torch.Tensor, out_class: torch.Tensor):
        self.this_class = this_class
        self.in_class = in_class
        self.out_class = out_class


class GorillaTripletDataPoint():
    this_class: DataPoint
    in_class: DataPoint
    out_class: DataPoint

    def __init__(self, this_class: DataPoint, in_class: DataPoint, out_class: DataPoint):
        self.this_class = this_class
        self.in_class = in_class
        self.out_class = out_class
    
    def to_tensor(self) -> torch.Tensor:
        raise Exception("FOO")


class GorillaDataset(Dataset):
    def __init__(
            self,
            datapoints: List[DataPoint],
            percent_start: int,
            percent_end: int,
        ):
        total_datapoints = len(datapoints)

        start_index = int(total_datapoints * percent_start / 100)
        end_index = int(total_datapoints * percent_end / 100) - 1
        self.datapoints = datapoints[start_index:end_index]

        # Group by gorilla-id
        self.datapoints_by_gorilla_id = {}

        for dp in self.datapoints:
            id = dp.gorilla_id
            if id not in datapoints:
                self.datapoints_by_gorilla_id[id] = []

            self.datapoints_by_gorilla_id[id].append(dp)
                
        self.dataset_indices_used_for_triplets = {}

        print(f"GORILLA_DATA: Using data from index {start_index} to {end_index}, covering {end_index - start_index}, total {total_datapoints} DP.")


    def get_triplet(self, idx: int) -> GorillaTripletDataPoint:
        base_datapoint = self.datapoints[idx]
        gorilla_id = base_datapoint.gorilla_id

        # Get random element from the same class.
        samples_in_this_class  = len(self.datapoints_by_gorilla_id[gorilla_id])
        if samples_in_this_class == 1:
            raise Exception("Ur dataset is fucked, cannot get other in-class element")

        in_class_index = idx

        while in_class_index == idx:
            in_class_index = random.randint(0, samples_in_this_class - 1)

        out_class_datapoint_index  = -1
        out_class_gorilla_id = gorilla_id
        while out_class_gorilla_id == gorilla_id:
            out_class_key_index = random.randint(0, len(self.datapoints_by_gorilla_id))
            random_out_class = list(self.datapoints_by_gorilla_id.keys())[out_class_key_index]

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
        # tensor_image, tensor_one_hot = read_image(complete_path_image, label_mapping)

        # files.append(tensor_image)
        # labels.append(tensor_one_hot)

        # return self.data[idx], self.labels[idx]
        return self.get_triplet(idx)
