import torch
from typing import Tuple
from . import utils, datapoint


class GorillaTripletTensor():
    this_class: torch.Tensor
    in_class: torch.Tensor
    out_class: torch.Tensor

    def __init__(
            self,
            this_class: torch.Tensor,
            in_class: torch.Tensor,
            out_class: torch.Tensor,
    ):
        self.this_class = this_class
        self.in_class = in_class
        self.out_class = out_class

    def to_triple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.this_class, self.in_class, self.out_class


class GorillaTripletDataPoint():
    this_class: datapoint.SPACDataPoint
    in_class: datapoint.SPACDataPoint
    out_class: datapoint.SPACDataPoint

    def __init__(
            self,
            this_class: datapoint.SPACDataPoint,
            in_class: datapoint.SPACDataPoint,
            out_class: datapoint.SPACDataPoint,
    ):
        self.this_class = this_class
        self.in_class = in_class
        self.out_class = out_class

    def to_tensor(self) -> GorillaTripletTensor:

        return GorillaTripletTensor(
            this_class=utils.read_image_to_tensor(self.this_class.filepath),
            in_class=utils.read_image_to_tensor(self.in_class.filepath),
            out_class=utils.read_image_to_tensor(self.out_class.filepath),
        )
