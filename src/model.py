import torch.nn as nn
from transformers import BertModel


class GBVModel (nn.Module):
    def __init__(self, num_labels):
        super(GBVModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased' , return_dict = False)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
       
        _,outpooled= self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
               
        )
        outpooled = self.dropout(outpooled)
        logits = self.out(outpooled)
        return logits 


