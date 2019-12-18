import boto3
import os
import pandas as pd
import time


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

print('one')

# Initialize standard VGG16 Model without FC layer
# Add custom Flatten and Dense layers to match output and input shapes

df = pd.read_excel(r'pgrfilestore.xlsx')
prefixes = list(df['Ulb Code'])
suffixes = list(df['File Storeid'])
for i in range(0, len(suffixes)):
    suffixes[i] = suffixes[i].strip()
complaint = list(df['Complaint Type'])
for i in range(0, len(complaint)):
    complaint[i] = complaint[i].strip()

print('two')

# Modify function to get object
# for objs in get_matching_s3_objects(
#         bucket=bucket_name,
#         prefix='1147/',
#         suffix='2cf305cc-2f06-4933-bc9d-6e57fb416fbc'
#         ):
#     all_objects.append(objs)

file_names = []

for i in range(0, 50):
    try:
        if " " in complaint[i]:
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
    except Exception:
        print(Exception)
