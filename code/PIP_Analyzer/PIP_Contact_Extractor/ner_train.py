import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from simpletransformers.ner import NERModel, NERArgs
import torch 

torch.cuda.empty_cache() 


logging.basicConfig(level=logging.INFO,
                    filename='ner.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# stopwords = [line.strip() for line in open("../data/stopwords.txt", 'r').readlines()] 
# all_data = pd.read_csv('../../data/ner_label.csv')
# all_data.columns = ['sentence_id','words', 'labels']
# all_data['words'].fillna(value='', inplace=True)
## all_data = all_data[~all_data['words'].isin(stopwords)]

train_data = pd.read_csv('../../data/groundtruth_dataset/Contact_extractor/train_ner.csv')
train_data['words'].fillna(value='', inplace=True)

eval_data = pd.read_csv('../../data/groundtruth_dataset/Contact_extractor/eval_ner.csv')
eval_data['words'].fillna(value='', inplace=True)

# index = np.random.permutation(1000)
# train_size = int(0.8 * len(index))
# train_data_sent_id = index[:train_size]
# test_data_sent_id = index[train_size:]
# train_data = all_data[all_data['sentence_id'].isin(train_data_sent_id)]
# train_data.to_csv('../data/train_ner.csv',index=False)
# eval_data = all_data[all_data['sentence_id'].isin(test_data_sent_id)]
# eval_data.to_csv('../data/eval_ner.csv',index=False)


# Create a NERModel
labels = ['O', 'B-qq', 'B-wechat', 'I-wechat', 'B-tg', 'I-tg', 'B-others','I-others', 'I-qq']
model = NERModel(
    "roberta",
    "xlm-roberta-base",
    labels=labels,
    args={
        'evaluate_during_training': True,
        'eval_steps': 3, 
        'logging_steps':3,
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "output_dir": './ner_model',
    },
   
)
model.train_model(train_data, eval_data=eval_data,args={
        'max_seq_length': 280,
        'num_train_epochs': 15, 
        'early_stopping_patience': 3,
        'early_stopping_threshold': 0.001}
        )

result, model_outputs, predictions = model.eval_model(eval_data)

print(result)