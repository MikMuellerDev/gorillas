import json
import os
from typing import List
from tqdm.notebook import tqdm
from . import utils
from concurrent.futures import ProcessPoolExecutor, as_completed


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
#
#
# Loader
#
#
#


def process_image(filename, image_dir_base):
    if not filename.endswith(".png"):
        return None
    complete_path_image = os.path.join(image_dir_base, filename)
    dp = utils.datapoint_from_path(complete_path_image, True)
    return dp


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
    
    
    print("[LOAD] Start indexing...")
    directory = os.fsencode(image_dir)
    files = [f for f in os.listdir(directory) if os.fsdecode(f).endswith(".png")]

    datapoints = []
    with ProcessPoolExecutor(max_workers=48) as executor:
        futures = [executor.submit(process_image, os.fsdecode(f), image_dir) for f in files]
        for f in tqdm(as_completed(futures), total=len(futures)):
            dp = f.result()
            if dp is not None:
                datapoints.append(dp)
    

    with open(utils.DP_DB_JSON_PATH, "w", encoding="utf-8") as f:
        dp_dicts = [dp.to_dict() for dp in datapoints]
        json.dump(dp_dicts, f, indent=4, ensure_ascii=False)

    print(f"[LOAD] Saved {len(datapoints)} datapoints to '{utils.DP_DB_JSON_PATH}'")
    return datapoints