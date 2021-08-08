import scipy.io
import csv
import glob
import os

mat = scipy.io.loadmat('data/raw_data/Annotations/Airplanes_Side_2/annotation_0001.mat')

# print(mat)
# print(mat['box_coord'])
# print(mat['box_coord'][0][0])
# print(mat['box_coord'][0][1])
# print(mat['box_coord'][0][2])
# print(mat['box_coord'][0][3])

f = csv.writer(open('data/annotation.csv', "w", newline=''))
f.writerow(['image_name', 'x1', 'y1', 'x2', 'y2'])

for file in glob.glob('data/raw_data/Annotations/Airplanes_Side_2/*.mat'):

    file_name = os.path.splitext(os.path.basename(file))[0]

    image_name = 'image_' + file_name.split('_')[1] + '.jpg'

    mat = scipy.io.loadmat(file)
    f.writerow([image_name, mat['box_coord'][0][2], mat['box_coord'][0][0], mat['box_coord'][0][3], mat['box_coord'][0][1]])
