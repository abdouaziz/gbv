import torch
import torch.nn as nn
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataset import VBGDataloader
from model import GBVModel
from utils import yield_optimizer 
import argparse
from transformers import get_linear_schedule_with_warmup
from engine import train_step , test
from tqdm import tqdm


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
    df = pd.read_csv("../input/Mytrain.csv")

    train_data , validation_data = train_test_split(df, test_size=0.2)


    train_loader = VBGDataloader(train_data, batch_size = args.batch_size, max_len = 100)
    test_loader = VBGDataloader(validation_data, batch_size = args.batch_size, max_len = 100)

    model = GBVModel(num_labels =5)
    model.to(device)
    optimizer = yield_optimizer(model)
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=40
        )

    for epoch in range(args.epochs):
        scheduler.step()
        train_step(model, train_loader, optimizer, epoch, device , scheduler)
        test(model, test_loader, device)
        torch.save(model.state_dict(), "../model/GBVModel.bin") 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labels', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=4)
  
    args = parser.parse_args()

    train(args)
    








 





       


 