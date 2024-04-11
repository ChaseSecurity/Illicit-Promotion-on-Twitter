'''
    note this script is for you to train, including random sample from all labeled data, if you want to reproduce our experiment, please directly read train/eval data to test
'''

import json
import re
import emoji
import random
import os
import pandas as pd
import seaborn as sns
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import numpy as np

def read_data(filepath):
    data = []
    with open(filepath,'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


re_url ='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
english_pattern = re.compile(r'[a-zA-Z]+')


def preprocess(data):
    data = re.sub(re_url,'',data)
    data = emoji.demojize(data)
    # data = re.sub(':\S+?:', ' ', data) 
    data = re.sub(r'[^\w\s]', '', data)
    # data = english_pattern.sub('', data)

    return data.replace('\n','')

def write2Json(data, filename):
    with open(filename,'w') as f:
        for line in data:
            f.write(json.dumps(line))
            f.write('\n')


path = "/data/xxx/media/"
file_names = os.listdir(path)
zero_size_file_names = [name for name in file_names if os.path.getsize(os.path.join(path,name)) == 0]
data = read_data('../../data/groundtruth_dataset/Multiclass_PIP_Classifier/all_labeled_data.json')
data = [each for each in data if each['labels']==1]

multi_class_data_dict = {}
label_list = list(set([each['subcategory'] for each in data]))
label_map_dict = {}

for idx,each in enumerate(label_list):
    label_map_dict[each] = str(idx)

idx2label = dict(zip(label_map_dict.values(), label_map_dict.keys()))

with open('../../data/groundtruth_dataset/Multiclass_PIP_Classifier/subcategory_label_mapping.json','w') as f:
    json.dump(label_map_dict, f)

for each in data:
    if each['labels'] == 1:
        if each['subcategory'] not in multi_class_data_dict.keys():
            multi_class_data_dict[each['subcategory']] = []
        # images = [tmp.replace('.mp4','_1.jpg') for tmp in each['media'] if tmp.endswith('.png') or tmp.endswith('.jpg') or tmp.endswith('.mp4')]
        # images = [tmp for tmp in images if tmp not in zero_size_file_names and tmp.replace('_1.jpg','.mp4') not in zero_size_file_names]
        # images = [tmp for tmp in images if tmp in file_names]
        tmp_dict = {}
        tmp_dict['ori_text'] = each['text']
        tmp_dict['text'] = preprocess(each['text'])
        tmp_dict['subcategory'] = each['subcategory']
        # tmp_dict['link'] = each['Link']
        # tmp_dict['media'] = each['media']
        # tmp_dict['images'] = images[0]
        tmp_dict['labels'] = label_map_dict[tmp_dict['subcategory']] 
        multi_class_data_dict[each['subcategory']].append(tmp_dict)
multi_class_data_train= []
multi_class_data_eval = []

for key,list_ in multi_class_data_dict.items():
    random.shuffle(list_)
    multi_class_data_train.extend(list_[:int(0.8*len(list_))])
    multi_class_data_eval.extend(list_[int(0.8*len(list_)):])


write2Json(multi_class_data_train, '../../data/groundtruth_dataset/Multiclass_PIP_Classifier/subcategory_data_reprocessed_train.json')
write2Json(multi_class_data_eval, '../data/groundtruth_dataset/Multiclass_PIP_Classifier/subcategory_data_reprocessed_eval.json')

random.shuffle(multi_class_data_train)
random.shuffle(multi_class_data_eval)


df_train = pd.DataFrame(multi_class_data_train)
df_eval = pd.DataFrame(multi_class_data_eval)


# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=15,  overwrite_output_dir=True)

# Create a ClassificationModel
text_model = ClassificationModel(
    'bert',
    'bert-base-multilingual-cased',
    num_labels=11,
    args=model_args
) 

df_train['labels'] = df_train['labels'].astype(int)
text_model.train_model(df_train,output_dir="text_cybercrime_subcategories")


