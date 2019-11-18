import numpy as np
from sklearn.neighbors import KNeighborsClassifier

file_names = []
image_vectors = []
garbage_file_names = []
garbage_image_vectors = []
tree_image_vectors = []
toilet_image_vectors = []
pothole_image_vectors = []
misc_image_vectors = []

with open('file_names.csv', 'r') as f:
    for line in f:
        file_names.append(line)

with open('image_vectors.csv', 'r') as f:
    for line in f:
        image_vector = line.split(',')
        for i in range(0, len(image_vector)):
            image_vector[i] = float(image_vector[i])
        image_vector = np.asarray(image_vector)
        image_vectors.append(image_vector)

with open('garbage_file_names.csv', 'r') as f:
    for line in f:
        garbage_file_names.append(line)

with open('garbage_image_vectors.csv', 'r') as f:
    for line in f:
        image_vector = line.split(',')
        for i in range(0, len(image_vector)):
            image_vector[i] = float(image_vector[i])
        image_vector = np.asarray(image_vector)
        garbage_image_vectors.append(image_vector)

with open('tree_image_vectors.csv', 'r') as f:
    for line in f:
        image_vector = line.split(',')
        for i in range(0, len(image_vector)):
            image_vector[i] = float(image_vector[i])
        image_vector = np.asarray(image_vector)
        tree_image_vectors.append(image_vector)

with open('toilet_image_vectors.csv', 'r') as f:
    for line in f:
        image_vector = line.split(',')
        for i in range(0, len(image_vector)):
            image_vector[i] = float(image_vector[i])
        image_vector = np.asarray(image_vector)
        toilet_image_vectors.append(image_vector)

with open('pothole_image_vectors.csv', 'r') as f:
    for line in f:
        image_vector = line.split(',')
        for i in range(0, len(image_vector)):
            image_vector[i] = float(image_vector[i])
        image_vector = np.asarray(image_vector)
        pothole_image_vectors.append(image_vector)

with open('random_image_vectors.csv', 'r') as f:
    for line in f:
        image_vector = line.split(',')
        for i in range(0, len(image_vector)):
            image_vector[i] = float(image_vector[i])
        image_vector = np.asarray(image_vector)
        misc_image_vectors.append(image_vector)

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='minkowski',
    algorithm='auto',
    p=2)

y_test = []
y_train = []
print("Training set: 4000 images, 700 images per category, top 5 complaints")
print("Testing set: 500 images, 100 per category")
image_vectors_training = []
image_vectors_testing = []

for i in range(0, 700):
    image_vectors_training.append(image_vectors[i])
    y_train.append(1)
for i in range(700, 800):
    image_vectors_testing.append(image_vectors[i])
    y_test.append(1)
for i in range(0, 700):
    image_vectors_training.append(garbage_image_vectors[i])
    y_train.append(0)
for i in range(700, 800):
    image_vectors_testing.append(garbage_image_vectors[i])
    y_test.append(0)
for i in range(0, 700):
    image_vectors_training.append(tree_image_vectors[i])
    y_train.append(2)
for i in range(700, 800):
    image_vectors_testing.append(tree_image_vectors[i])
    y_test.append(2)
for i in range(0, 700):
    image_vectors_training.append(toilet_image_vectors[i])
    y_train.append(3)
for i in range(700, 800):
    image_vectors_testing.append(toilet_image_vectors[i])
    y_test.append(3)
for i in range(0, 700):
    image_vectors_training.append(pothole_image_vectors[i])
    y_train.append(4)
for i in range(700, 800):
    image_vectors_testing.append(pothole_image_vectors[i])
    y_test.append(4)
for i in range(0, 500):
    image_vectors_training.append(misc_image_vectors[i])
    y_train.append(5)

X_train = image_vectors_training

knn.fit(X_train, y_train)
accuracy = knn.score(image_vectors_testing, y_test)*100
print("Accuracy on testing set: "+str(accuracy))

