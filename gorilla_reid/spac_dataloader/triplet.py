import torch
from datapoint import SPACDataPoint
from typing import Tuple
from utils import read_image_to_tensor


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
    this_class: SPACDataPoint
    in_class: SPACDataPoint
    out_class: SPACDataPoint

    def __init__(
            self,
            this_class: SPACDataPoint,
            in_class: SPACDataPoint,
            out_class: SPACDataPoint,
    ):
        self.this_class = this_class
        self.in_class = in_class
        self.out_class = out_class

    def to_tensor(self) -> GorillaTripletTensor:

        return GorillaTripletTensor(
            this_class=read_image_to_tensor(self.this_class.filepath),
            in_class=read_image_to_tensor(self.in_class.filepath),
            out_class=read_image_to_tensor(self.out_class.filepath),
        )
