import torch
from torchvision.io import decode_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

from torchvision.datasets.mnist import MNIST


dataset = MNIST(root='data', download=True, transform=transforms.ToTensor())



# class Dataset(Dataset):
#     def __init__(self, dir_raw):
        
#         #dir_raw = "/scratch2/gorillawatch/data/detection_datasets/body_detection_gorilla/val/"
#         # dir_raw = "/scratch2/gorillawatch/berlin_zoo_data/val/"
#         directory = os.fsencode(dir_raw)

#         #limit = 20

#         files = []
#         labels = []
#         for [idx, file] in enumerate(os.listdir(directory)):
#             # if idx >= limit:
#             #     break

#             filename = os.fsdecode(file)
#             if filename.endswith(".png"): 
#                 # filename_txt_complete = os.path.join(dir_raw, filename.replace(".png", ".txt"))

#                 complete_path_image = os.path.join(dir_raw, filename)
#                 img = Image.open(complete_path_image)
#                 rez_tr = transforms.Resize((224, 224))
#                 rez_img = rez_tr(img)
#                 transform_to_tensor = transforms.Compose([transforms.PILToTensor()])
#                 tensor_image = transform_to_tensor(rez_img)

#                 label = filename.split("_")[0]
#                 # print(label)
#                 # with open(filename_txt_complete, 'r') as txtfile:
#                 #     label = txtfile.read()

#                 files.append(tensor_image)
#                 labels.append(label)
#                 continue
#             else:
#                 continue

#         for [idx, file] in enumerate(files):
#             pass

        
#         # END IMAGE LOAD


#         # self.root_dir = root_dir
#         # self.transform = transform
#         # labels = [i for i in range (0, 51)]

#         # data = [torch.arange(50) for i in range(0, 51)]

#         self.labels = labels
#         self.data = files

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]


# dir_raw = "/scratch2/gorillawatch/data/SPAC_face_images/face_images/"
# dataset = ExampleDataset(dir_raw=dir_raw)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for b in dataloader:
#     images, labels = b
#     print(images, labels)
#     print("####")



