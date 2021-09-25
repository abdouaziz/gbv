import json
import torch.nn as nn
from transformers import BertModel


with open("config.json") as f :
    config = json.load(f)


class GBVClassifier(nn.Module):
    def __init__(self , n_classes):
        super(GBVClassifier,self).__init__()
        self.bert = BertModel.from_pretrained(config["BERT_MODEL"] , return_dict=False)
        self.drop=  nn.Dropout(p=0.9)
        self.out = nn.Linear(768 , n_classes)

    def forward(self , input_ids, attention_mask ,token_type_ids):
        
        _,sortie = self.bert(
                        input_ids , 
                        attention_mask ,
                        token_type_ids  )
        output = self.drop(sortie)
        output = self.out(output)

        return output