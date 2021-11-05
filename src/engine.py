import torch
import torch.nn as nn
from tqdm import tqdm
from utils import loss_fn

 

def train_step(model, train_loader, optimizer, epoch, device , scheduler):
   
    model.train()
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
      
        output = model(batch['ids'].to(device), batch['mask'].to(device), batch['token_type_ids'].to(device))
        loss = loss_fn(output, batch['labels'].to(device))

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        
    
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch['ids']), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



def test(model, test_loader, device):
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            output = model(batch['ids'].to(device), batch['mask'].to(device), batch['token_type_ids'].to(device))
            test_loss += loss_fn(output, batch['labels'].to(device)).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(batch['labels'].to(device)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(f"step {batch_idx}")


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

