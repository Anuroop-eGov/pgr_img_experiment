import boto3
import os
import shutil
import pandas as pd
import time
import csv
from keras import Input
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# Predefined function that retrieves S3 objects by prefix/suffix
def get_matching_s3_objects(bucket, prefix="", suffix=""):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {'Bucket': bucket}
    if isinstance(prefix, str):
        prefixes = (prefix, )
    else:
        prefixes = prefix
    for key_prefix in prefixes:
        kwargs["Prefix"] = key_prefix
        for page in paginator.paginate(**kwargs):
            try:
                contents = page["Contents"]
            except KeyError:
                return
            for obj in contents:
                key = obj["Key"]
                if key.endswith(suffix):
                    yield obj


# Initialize S3 locator
bucket_name = 'ap-pgr-images'
s3 = boto3.client('s3')
source = boto3.resource('s3')
bucket = source.Bucket(bucket_name)

# Initialize standard VGG16 Model without FC layer
# Add custom Flatten and Dense layers to match output and input shapes
vgg_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(512, 512, 3)))
model = Sequential(name="VGG16_Preprocessor")
for layer in vgg_model.layers[:]:
    model.add(layer)
model.add(Flatten())
model.add(Dense(4096, input_shape=(131072,)))

df = pd.read_excel(r'pgrfilestore.xlsx')
prefixes = list(df['Ulb Code'])
suffixes = list(df['File Storeid'])
for i in range(0, len(suffixes)):
    suffixes[i] = suffixes[i].strip()
complaint = list(df['Complaint Type'])
for i in range(0, len(complaint)):
    complaint[i] = complaint[i].strip()

# Modify function to get object
# for objs in get_matching_s3_objects(
#         bucket=bucket_name,
#         prefix='1147/',
#         suffix='2cf305cc-2f06-4933-bc9d-6e57fb416fbc'
#         ):
#     all_objects.append(objs)

image_vectors = []
file_names = []

for i in range(0, len(prefixes)):
    try:
        if "toilet" in complaint[i]:
            file_names.append(str(prefixes[i])+'/'+suffixes[i])
            print("File name: "+str(prefixes[i])+'/'+suffixes[i]+'.jpg')
            start = time.time()
            for objs in get_matching_s3_objects(
                    bucket=bucket_name,
                    prefix=str(prefixes[i])+'/',
                    suffix=suffixes[i]
                    ):
                if os.path.isdir(str(prefixes[i])):
                    with open(
                            str(prefixes[i])+'/'+suffixes[i]+'.jpg',
                            'wb') as f:
                        s3.download_fileobj(bucket_name, objs['Key'], f)
                else:
                    os.mkdir(str(prefixes[i]))
                    with open(
                            str(prefixes[i])+'/'+suffixes[i]+'.jpg',
                            'wb') as f:
                        s3.download_fileobj(bucket_name, objs['Key'], f)
            end = time.time()
            print("Download Time : " + str(end - start))
        # Loading test image
            start = time.time()
            image = load_img(
                str(prefixes[i])+'/'+suffixes[i]+'.jpg',
                target_size=(512, 512))
            image = img_to_array(image)
            image = image.reshape((
                1,
                image.shape[0],
                image.shape[1],
                image.shape[2]))
            image = preprocess_input(image)
        # print(model.predict(image)[0])
            image_vectors.append(model.predict(image)[0])
            end = time.time()
            print("Vectorization Time : " + str(end - start))
            shutil.rmtree(str(prefixes[i]))
            with open("toilet_image_vectors.csv", "a") as f:
                wr = csv.writer(f)
                wr.writerow(model.predict(image)[0])
            with open("toilet_file_names.csv", "a") as f:
                wr = csv.writer(f)
                wr.writerow([str(prefixes[i])+'/'+suffixes[i]])
    except Exception:
        print(Exception)
