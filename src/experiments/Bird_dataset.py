import json
import os
import numpy as np

root = '/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/birds/CUB_200_2011/CUB_200_2011'
image_dictionary = {}
id_image_path = os.path.join(root,'images.txt')
k_shot = 5

# Image location
with open(id_image_path,'r') as f:
    for line in f.readlines():
        info = line.split()
        image_dictionary[info[0]] = []
        image_dictionary[info[0]].append(info[1])

# Image category label
image_label_path = os.path.join(root,'image_class_labels.txt')
with open(image_label_path,'r') as f1:
    for line in f1.readlines():
        info = line.split()
        # -1 so that the class label matches the index of the most probable label
        image_dictionary[info[0]].append(int(info[1])-1)

# Image auxiliary label
image_attribute_path = os.path.join(root, 'attributes', 'image_attribute_labels.txt')
with open(image_attribute_path,'r') as f2:
    # Every line contains <image_id> <attribute_id> <is_present> <certainty_id> <time>
    for line in f2.readlines():
        info = line.split()
        # If image <image_id> and only its class is in the dictionary
        if len(image_dictionary[info[0]]) == 2:
            # Add a field in the dictionary for binary labels
            image_dictionary[info[0]].append([])
        # Add the binary labels to the binary labels field
        image_dictionary[info[0]][2].append(int(info[2]))

# Directory to put all preprocessed data
outroot = os.path.join("/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/birds/CUB_200_2011/", 'preprocess_data')

# Training and test split
all_train_id_set = []
all_test_id_set = []
with open(os.path.join(root,'train_test_split.txt'),'r') as f4:
    for line in f4.readlines():
        info = line.split()
        if info[1] == '1':
            all_train_id_set.append(info[0])
        else:
            all_test_id_set.append(info[0])

print("Before shuffle:", all_test_id_set[0:12])
import random
random.shuffle(all_test_id_set)
print("After shuffle:", all_test_id_set[0:12])

# Split the test set into validation and test sets
valid_id_set = all_test_id_set[0:2897]
test_id_set = all_test_id_set[2897:5794]

# Modify the full train set to have only 5 labels per class
full_train_set = []
cate_recorder = np.zeros(200)

# Loop through all images in the original training set
for img_id in all_train_id_set:
    # Category for an image
    cate_for_img = image_dictionary[img_id][1]
    # If less than k_shot labels of this cat. are present in the full train set
    if cate_recorder[cate_for_img] < k_shot:
        # Save this image to the full train set
        full_train_set.append(img_id)
        # Record that an extra label for this cat. has been added to the full train set
        cate_recorder[cate_for_img] += 1
    else:
        # Save this image to the full train set
        full_train_set.append(img_id)
        # Set the label of this image to -1
        image_dictionary[img_id][1] = -1

# Save the dictionary that contains <location>, <label> and <binary_attributes> per image
with open( os.path.join(outroot,'image_dictionary.json'), 'w' ) as f3:
    json.dump(image_dictionary, f3)

# Save the full train set
with open(os.path.join(outroot, 'full_train_set.json'), 'w') as f:
    json.dump(full_train_set, f)

#with open( os.path.join(outroot,'full_train_set.json'), 'w' ) as f:
#    json.dump(all_train_id_set, f)

with open( os.path.join(outroot,'valid_set.json'), 'w' ) as f:
    json.dump(valid_id_set, f)

with open( os.path.join(outroot,'test_set.json'), 'w' ) as f:
    json.dump(test_id_set, f)

print("Train set length:", len(full_train_set))
print("Validation set length:", len(valid_id_set))
print("Test set length:", len(test_id_set))

# Split the training set into auxiliary and training sets
aux_id_set = full_train_set[0:256]
rest_train_set = full_train_set[256:5994]

print("Auxiliary set length:", len(aux_id_set))
print("Final train set length:", len(rest_train_set))

with open( os.path.join(outroot,'aux_set.json'), 'w' ) as f:
    json.dump(aux_id_set, f)

with open( os.path.join(outroot,'rest_train_set.json'), 'w' ) as f:
    json.dump(rest_train_set, f)
