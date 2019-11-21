import boto3
import pandas as pd
from kafka import KafkaProducer

# Initialize S3 locator
bucket_name = 'ap-pgr-images'
s3 = boto3.client('s3')
source = boto3.resource('s3')
bucket = source.Bucket(bucket_name)
producer = KafkaProducer()

df = pd.read_excel(r'PGR_FILESTORE.xlsx')
prefixes = list(df['Ulb Code'])
suffixes = list(df['File Storeid'])

for i in range(0, len(suffixes)):
    suffixes[i] = suffixes[i].strip()
    filestore_id = bytes(str(prefixes[i])+'/'+suffixes[i], encoding='utf-8')
    producer.send('filestore_ids', value=filestore_id)
    print("Sent filestore id: "+str(prefixes[i])+'/'+suffixes[i])
