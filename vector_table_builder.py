from kafka import KafkaConsumer
from json import loads
from csv import writer
import numpy as np

vector_consumer = KafkaConsumer(
                    'image_vectors',
                    group_id='table_builder',
                    auto_offset_reset='earliest',
                    value_deserializer=lambda x: loads(x.decode('utf-8')))

for i in vector_consumer:
    w = writer(open("vector_table.csv", "a"))
    w.writerow([i.key, np.array(i.value)])
