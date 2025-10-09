import json
from typing import List
import torch
from torchvision.io import decode_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import os
from tqdm.notebook import tqdm

SPAC_SYNTH_PATH = "./SPAC_SYNTHETIC"

def generate_synthetic_data():
    dir_raw = "/scratch2/gorillawatch/data/SPAC_face_images/face_images/"
    directory = os.fsencode(dir_raw)

    le = len(os.listdir(directory))
    pbar = tqdm(total=le, desc="Generating...")

    for [idx, file] in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            complete_path_image = os.path.join(dir_raw, filename)
            img = Image.open(complete_path_image)
            rez_tr = transforms.Resize((224, 224))
            rez_img = rez_tr(img)

            for i in range(0, 36):
                deg = i * 10
                rot_img = rez_img.rotate(deg)
                rot_img.save(os.path.join(SPAC_SYNTH_PATH, f"{filename}_rotated-{deg}DEG.png"))

            pbar.update(1)    # manually advance by 1 step

            continue
        else:
            continue

def build_label_index(dirs = ["/scratch2/gorillawatch/data/SPAC_face_images/face_images/", "./SPAC_SYNTHETIC"]) -> dict: 
    output_json = "label_db.json"

    try:
        with open(output_json, "r", encoding="utf-8") as f:
            print("Using label cache...")
            return json.loads(f.read())
    except Exception as e:
        print(f"EX: {e}")
        pass

    label_set = {}
    files = []
    labels = []

    total_files = 0
    for dir_raw in dirs:
        directory = os.fsencode(dir_raw)
        total_files += len(os.listdir(directory))

    pbar = tqdm(total=total_files, desc="(Label Cache) Indexing dataset...")

    for dir_raw in dirs:
        directory = os.fsencode(dir_raw)

        for [idx, file] in enumerate(os.listdir(directory)):
            filename = os.fsdecode(file)
            if filename.endswith(".png"): 
                complete_path_image = os.path.join(dir_raw, filename)
                img = Image.open(complete_path_image)
                rez_tr = transforms.Resize((224, 224))
                rez_img = rez_tr(img)
                transform_to_tensor = transforms.Compose([transforms.PILToTensor()])
                tensor_image = transform_to_tensor(rez_img)

                label = filename.split("_")[0]

                if label not in label_set:
                    label_set[label] = len(label_set)

                files.append(tensor_image)
                labels.append(label)

                pbar.update(1)
                continue
            else:
                continue
    
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(label_set, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(label_set)} labels to '{output_json}'")
    for k, v in label_set.items():
        print(f"{v} -> {k}")

    return label_set


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


def get_modified_filelist(base: str) -> List[str]:
    out = []

    spac = SPAC_SYNTH_PATH
    directory = os.fsencode(spac)
    for [idx, file] in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        
        original_name = filename.split("_rotated")[0]

        if original_name == base:
            out.append(filename)

    print(f"Found {len(out)} matching transformations for {base}")
    return out
            

class GorillaDataset(Dataset):
    def __init__(self, percent_start: int, percent_end: int, label_mapping: dict, include_transformed: bool):
        files = []
        labels = []

        total_files = 0

        modified_search_dir = ["./SPAC_SYNTHETIC"]

        dirs = dirs=["/scratch2/gorillawatch/data/SPAC_face_images/face_images/"]
        for dir_raw in dirs:
            directory = os.fsencode(dir_raw)
            total_files += len(os.listdir(directory))
 

        current_file_idx = 0

        start_index = int(total_files * percent_start / 100)
        end_index = int(total_files * percent_end / 100) - 1
        covering = end_index - start_index


        if include_transformed:
            covering *= 36

        print(f"GORILLA_DATA: Using data from index {start_index} to {end_index}, covering {covering}, total {total_files} samples.")
        pbar = tqdm(total=covering, desc="Loading Dataset...")

        for dir_raw in dirs:
            directory = os.fsencode(dir_raw)

            for [idx, file] in enumerate(os.listdir(directory)):
                current_file_idx += 1

                if current_file_idx < start_index:
                    continue
                if current_file_idx > end_index:
                    break


                filename = os.fsdecode(file)
                if filename.endswith(".png"): 
                    # Crunch base image.
                    complete_path_image = os.path.join(dir_raw, filename)
                    tensor_image, tensor_one_hot = read_image(complete_path_image, label_mapping)

                    files.append(tensor_image)
                    labels.append(tensor_one_hot)

                    # If we include the modified images, load them now.
                    if include_transformed:
                        files_a = get_modified_filelist(filename)
                        for mod_file in files_a:
                            #print(f"MOD: {mod_file}")
                            complete_path_image = os.path.join(SPAC_SYNTH_PATH, mod_file)
                            try:
                                tensor_image, tensor_one_hot = read_image(complete_path_image, label_mapping)

                                files.append(tensor_image)
                                labels.append(tensor_one_hot)

                                if include_transformed:
                                    pbar.update(1)
                            except Exception as e:
                                print(f"E: {e}")

                    pbar.update(1)

                    continue
                else:
                    continue


        self.data = files
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
