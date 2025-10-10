from typing import List


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
