import boto3
import os
import shutil
from json import dumps
from kafka import KafkaConsumer
from kafka import KafkaProducer
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
consumer = KafkaConsumer(
            'filestore_ids',
            group_id='vectorizer',
            auto_offset_reset='earliest')
producer = KafkaProducer(value_serializer=lambda x: dumps(x).encode('utf-8'))

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

# Modify function to get object
# for objs in get_matching_s3_objects(
#         bucket=bucket_name,
#         prefix='1147/',
#         suffix='2cf305cc-2f06-4933-bc9d-6e57fb416fbc'
#         ):
#     all_objects.append(objs)

for i in consumer:
    filestore_id = i.value.decode("utf-8")
    prefix = filestore_id[0:4]
    print("Received filestore id: "+filestore_id)
    for objs in get_matching_s3_objects(
            bucket=bucket_name,
            prefix=filestore_id[0:4],
            suffix=filestore_id[5:]
            ):
        if os.path.isdir(prefix):
            with open(filestore_id+'.jpg', 'wb') as f:
                s3.download_fileobj(bucket_name, objs['Key'], f)
        else:
            os.mkdir(prefix)
            with open(filestore_id+'.jpg', 'wb') as f:
                s3.download_fileobj(bucket_name, objs['Key'], f)
    image = load_img(
        filestore_id+'.jpg',
        target_size=(512, 512))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    # print(model.predict(image)[0])
    image_vector = model.predict(image)[0]
    os.remove(filestore_id+'.jpg')
    shutil.rmtree(prefix)
    producer.send(
        'image_vectors', {filestore_id: image_vector.tolist()})
    print(filestore_id+" vectorized!")
    consumer.commit()
