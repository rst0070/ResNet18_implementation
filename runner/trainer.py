from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import TrainDataset
from arguments import get_args
import wandb

sys_args, exp_args = get_args()

GPU = sys_args['gpu']
NUM_WORKERS = sys_args['num_workers']

class Trainer():
    def __init__(self, model, dataset:TrainDataset, optimizer, batch_size, pre_amphasis, log_melspec):
        self.pre_amphasis = pre_amphasis
        self.log_melspec = log_melspec
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = NUM_WORKERS)
        self.loss_function = nn.CrossEntropyLoss().to(GPU)
        self.optimizer = optimizer

    def train(self):
        self.model.train()
        
        itered = 0
        loss_sum = 0
        
        for X, y in tqdm(self.dataloader, desc = "training"):
            itered = itered + 1
            self.optimizer.zero_grad()
            
            X = X.to(GPU)
            y = y.to(GPU)
            
            X = self.pre_amphasis(X)
            X = self.log_melspec(X)
            pred = self.model(X, is_test = False)
            loss = self.loss_function(pred,y)
            loss_sum += loss
            
            loss.backward()
            self.optimizer.step()
            
            if itered == 50:
                wandb.log({'Loss':loss_sum / float(itered)})
                itered = 0
                loss_sum = 0
                
        wandb.log({'Loss':loss_sum / float(itered)})
                
                