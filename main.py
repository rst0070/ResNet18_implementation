import arguments
from model import ResNet_18
import torch
from data import dataset
from data.feature import AudioPreEmphasis, LogMelSpec
import runner.trainer as trainer
import runner.tester as tester
import os
import wandb
import random
import torch.backends.cudnn as cudnn
import numpy as np

class Main:
    def __init__(self):
        sys_args, exp_args = arguments.get_args()
        
        # seed 설정
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(0)
        
        if sys_args['wandb_disabled']: # arguments에 wandb 설정확인(wandb loggin 끄는 코드)
            os.system("wandb disabled")
            
        os.system(f"wandb login {sys_args['wandb_key']}") # arguments의 sys_args['wandb_key']에 자신의 key 입력 필요
        wandb.init(
            project = sys_args['wandb_project'],
            entity = sys_args['wandb_entity'],
            name = sys_args['wandb_name']
        )
        
        
        GPU = sys_args['gpu']
        
        self.max_epoch = exp_args['epoch']
        
        self.pre_amphasis = AudioPreEmphasis().to(GPU)
        self.log_melspec = LogMelSpec(exp_args=exp_args).to(GPU)
        self.model = ResNet_18().to(GPU)
        
        # optimizer 정의
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = exp_args['lr'],
            weight_decay = exp_args['weight_decay']
        )
        
        # learning rate가 epoch마다 0.95%씩 감소하도록 설정
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.97
        )
        
        train_dataset = dataset.TrainDataset(annotation_file_path = sys_args['path_train_label'], data_dir = sys_args['path_train'])
        test_dataset = dataset.TestDataset(annotation_file_path = sys_args['path_test_label'], data_dir = sys_args['path_test'])
        
        self.trainer = trainer.Trainer(model = self.model, dataset = train_dataset, optimizer = optimizer, batch_size = exp_args['batch_size'], pre_amphasis=self.pre_amphasis, log_melspec=self.log_melspec)
        self.tester = tester.Tester(model = self.model, dataset = test_dataset, batch_size=exp_args['batch_size'], pre_amphasis=self.pre_amphasis, log_melspec=self.log_melspec)
        
    def start(self):
        
        min_eer = 100
        model_state = None
        for epoch in range(1, self.max_epoch + 1):
            
            self.trainer.train()
            
            eer = self.tester.test(epoch = epoch)
            
            if eer < min_eer:
                min_eer = eer
                model_state = self.model.state_dict()

            self.lr_scheduler.step()
            
        file_name = f"eer_{eer}.pth"
        torch.save(model_state, file_name)
        wandb.save(file_name)
        
            

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method("spawn")
    program = Main()
    program.start()
    