import json
import os
from typing import List
from tqdm.notebook import tqdm
from . import utils


class SPACDataPoint():
    gorilla_id: str
    camera: str
    date: str
    # //////
    filepath: str
    transformations: List[str]

    def __init__(
            self,
            gorilla_id: str,
            camera: str,
            date: str,
            filepath: str,
            transformations: List[str],
    ):
        self.gorilla_id = gorilla_id
        self.camera = camera
        self.date = date
        self.filepath = filepath
        self.transformations = transformations

    def to_dict(self):
        return self.__dict__


def spac_datapoint_from_dict(dictionary) -> SPACDataPoint:
    _self = SPACDataPoint(
        gorilla_id="",
        camera="",
        date="",
        filepath="",
        transformations=[],
    )
    for k, v in dictionary.items():
        setattr(_self, k, v)
    return _self



#
# Loader
#


def load_datapoints(image_dir=utils.DEFAULT_SPAC_IMAGE_DIR) -> List[SPACDataPoint]:
    """
    Loads datapoint index from disk.
    If the cache file does not exist, this can take a while because it needs
    to scan the entire file structure.
    """
    try:
        with open(utils.DP_DB_JSON_PATH, "r", encoding="utf-8") as f:
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

    stepsize = 10
    pbar = tqdm(total=total_files, desc="[LOAD] Indexing...")

    # Multithreaded

    for [idx, file] in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            complete_path_image = os.path.join(image_dir, filename)
            dp = utils.datapoint_from_path(complete_path_image, True)
            datapoints.append(dp)
            if idx % stepsize == 0:
                pbar.update(stepsize)
            continue
        else:
            continue

    with open(utils.DP_DB_JSON_PATH, "w", encoding="utf-8") as f:
        dp_dicts = [dp.to_dict() for dp in datapoints]
        json.dump(dp_dicts, f, indent=4, ensure_ascii=False)

    print(f"[LOAD] Saved {len(datapoints)} datapoints to '{utils.DP_DB_JSON_PATH}'")
    return datapoints