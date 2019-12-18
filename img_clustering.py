import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.mixture import GaussianMixture
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import GaussianNB

# file_names = []
image_vectors = []
# garbage_file_names = []
garbage_image_vectors = []
tree_image_vectors = []
toilet_image_vectors = []
pothole_image_vectors = []
misc_image_vectors = []

# with open('file_names.csv', 'r') as f:
#     for line in f:
#         file_names.append(line)

with open('image_vectors.csv', 'r') as f:
    for line in f:
        image_vector = line.split(',')
        for i in range(0, len(image_vector)):
            image_vector[i] = float(image_vector[i])
        image_vector = np.asarray(image_vector)
        image_vectors.append(image_vector)

# with open('garbage_file_names.csv', 'r') as f:
#     for line in f:
#         garbage_file_names.append(line)

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
    n_neighbors=50,
    weights='distance',
    metric='minkowski',
    algorithm='auto',
    p=2)

# knn = KMeans(n_clusters=6, random_state=0)

# knn = GaussianMixture(n_components=6)

# knn = MLPClassifier(
#     hidden_layer_sizes=(256, 256, 128),
#     activation='relu',
#     solver='adam',
#     max_iter=1000, random_state=1)

# knn = GaussianNB()

y_test = []
y_train = []
print("Loading Top 5 categories, 5300 image dataset.")
image_vectors_training = []
image_vectors_testing = []

for i in range(0, 800):
    image_vectors_training.append(garbage_image_vectors[i])
    y_train.append(0)
for i in range(800, 900):
    image_vectors_testing.append(garbage_image_vectors[i])
    y_test.append(0)
for i in range(0, 800):
    image_vectors_training.append(image_vectors[i])
    y_train.append(1)
for i in range(800, 900):
    image_vectors_testing.append(image_vectors[i])
    y_test.append(1)
for i in range(0, 800):
    image_vectors_training.append(tree_image_vectors[i])
    y_train.append(2)
for i in range(800, 900):
    image_vectors_testing.append(tree_image_vectors[i])
    y_test.append(2)
for i in range(0, 800):
    image_vectors_training.append(toilet_image_vectors[i])
    y_train.append(3)
for i in range(800, 900):
    image_vectors_testing.append(toilet_image_vectors[i])
    y_test.append(3)
for i in range(0, 800):
    image_vectors_training.append(pothole_image_vectors[i])
    y_train.append(4)
for i in range(800, 900):
    image_vectors_testing.append(pothole_image_vectors[i])
    y_test.append(4)
for i in range(0, 800):
    image_vectors_training.append(misc_image_vectors[i])
    y_train.append(5)

X_train = image_vectors_training

knn.fit(X_train, y_train)
# print(knn.cluster_centers_)
# print(X_train[0])
# for i in knn.cluster_centers_:
#     print(i)
#     print(y_train[[x.tolist() for x in X_train].index(i.tolist())])
# print(knn.predict(image_vectors_testing))
# print(knn.score(image_vectors_testing, y_test))
# print(knn.n_iter_)
# print(knn.loss_)


def guess_image(test_vector):
    prediction = knn.predict_proba(test_vector)
    print(prediction)
    return prediction
