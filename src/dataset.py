
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer




class GBVDataset(Dataset):
    """Custom Dataset for the VBG dataset"""

    def __init__(self, texts, labels,  max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        tokenized_text = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )

        return {
            'ids': tokenized_text["input_ids"].flatten(),
            'mask': tokenized_text["attention_mask"].flatten(),
            'token_type_ids': tokenized_text["token_type_ids"].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)

        }


def VBGDataloader(df ,batch_size, max_len):
    
    train_dataset = GBVDataset(
                            texts=df.tweet.values,
                            labels=df.label.values,
                            max_len=max_len
                        )

    train_loader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                shuffle=False
                              
                            )

    return train_loader
 