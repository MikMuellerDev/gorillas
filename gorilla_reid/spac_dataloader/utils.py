import os
from PIL import Image
import torchvision.transforms as transforms
from datapoint import SPACDataPoint
from typing import List


def rfile(path: str) -> str:
    """
    Returns an absolute filepath given a relative path (to the current PY file)
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(file_dir, path)


SPAC_SYNTH_PATH = rfile("../../SPAC_SYNTHETIC")
DP_DB_JSON_PATH = rfile("../../json/dp_db.json")
DEFAULT_SPAC_IMAGE_DIR = "/scratch2/gorillawatch/data/SPAC_face_images/face_images/"


def read_image_to_tensor(abspath: str, image_dim=(224, 224)):
    img = Image.open(abspath)
    rez_tr = transforms.Resize(image_dim)
    rez_img = rez_tr(img)
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    tensor_image = transform_to_tensor(rez_img)
    return tensor_image


def datapoint_from_path(abspath: str, with_transformed_filelist: bool):
    filename = os.basename(abspath)
    path_segments = filename.split("_")
    gorilla_id = path_segments[0]
    camera_id = path_segments[1]
    date = path_segments[2]

    transformations = []
    if with_transformed_filelist:
        transformations = get_modified_filelist(filename)

    dp = SPACDataPoint(
        gorilla_id=gorilla_id,
        camera=camera_id,
        date=date,
        filepath=abspath,
        transformations=transformations
    )
    return dp


def get_modified_filelist(base: str) -> List[str]:
    out = []

    spac = SPAC_SYNTH_PATH
    directory = os.fsencode(spac)
    for [idx, file] in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        original_name = filename.split("_rotated")[0]

        if original_name == base:
            out.append(filename)
    return out
