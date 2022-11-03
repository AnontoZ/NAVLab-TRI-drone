import cv2
from os import listdir
from os.path import isfile, join
jpg_folder = "data/tag_comparison/jpg/"
onlyfiles = [f for f in listdir(jpg_folder) if isfile(join(jpg_folder, f))]
print(onlyfiles)

for file in onlyfiles:
    filepath = jpg_folder+file
    img = cv2.imread(filepath)
    cv2.imwrite("data/tag_comparison/tag16h5_comparison/"+"".join(file.split(".")[:-1])+".png", img)