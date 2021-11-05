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
import logging

logger = logging.getLogger(__name__)

# Parsing input arguments
def parse_args():

    parser = argparse.ArgumentParser(description="Pretrain transformers model on a text classification task , GBV")
    parser.add_argument(
        "--num_labels",
        type=int,
        default=5,
        help="The number of labels that we will classify the input giving.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="The batch_size of training the model",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=4,
        help="The number total of epochs that we will train the model",
    ) 

    args = parser.parse_args()

    if args.num_labels is None:
        raise ValueError("Need the number of labels to traint the model")

    if args.batch_size is None:
        raise ValueError("Need the number of batch to traint the model")

    if args.epochs is None:
        raise ValueError("Need the number of epochs to traint the model")


    return args



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
    df = pd.read_csv("../input/Mytrain.csv")

    train_data , validation_data = train_test_split(df, test_size=0.2)


    train_loader = VBGDataloader(train_data, batch_size = args.batch_size, max_len = 100)
    test_loader = VBGDataloader(validation_data, batch_size = args.batch_size, max_len = 100)


    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Training batch size  = {args.batch_size}")
    logger.info(f"  Total of labels Trainig = {args.num_labels}")


    model = GBVModel(num_labels =args.num_labels)
    model.to(device)
    optimizer = yield_optimizer(model)
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=40
        )
    for epoch in range(args.epochs):
        
        train_step(model, train_loader, optimizer, epoch, device , scheduler)
        test(model, test_loader, device)
        torch.save(model.state_dict(), "../model/GBVModel.bin") 



if __name__ == '__main__':

    main()
    








 





       


 