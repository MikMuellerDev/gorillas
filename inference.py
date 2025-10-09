from ultralytics import YOLO
import os

# Load a model
model = YOLO("./YOLOv11_WB/experiment_112/weights/best.pt")

# Run batched inference on a list of images

dir_raw = "/scratch2/gorillawatch/data/detection_datasets/body_detection_gorilla/val/"
# dir_raw = "/scratch2/gorillawatch/berlin_zoo_data/val/"
directory = os.fsencode(dir_raw)

limit = 20

files = []
for [idx, file] in enumerate(os.listdir(directory)):
    if idx >= limit:
        break

    filename = os.fsdecode(file)
    if filename.endswith(".png"): 
        files.append(os.path.join(dir_raw, filename))
        continue
    else:
        continue

for [idx, file] in enumerate(files):
    print(file)
    # continue
    results = model([file])  # return a list of Results objects

    output = f"{idx}_result_gorilla.jpg"

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.save(filename=output)  # save to disk
        # result.show()  # display to screen
        
    os.system(f"code {output}")