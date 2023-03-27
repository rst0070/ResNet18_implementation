import torch
import torchaudio
import os
import random

class MusanNoise:
    
    categories = ['noise', 'music', 'speech']
    files = {
        'noise' : [], 'music' : [], 'speech' : []
    }
    
    def __init__(self, root_path:str):
        """_summary_

        Args:
            root_path (str): root path of musan dataset
        """
        
        for dir_path, _, files in os.walk(root_path):
            category = None
            for c in self.categories:
                if c in dir_path:
                    category = c
                    break
            if category is not None:
                for file in files:
                    if '.wav' in file:
                        self.files[category].append(os.path.join(dir_path, file))                
        
    
    def __call__(self, x, snr):
        assert snr > 0 and len(x.shape) == 1
        
        category = random.choice(self.categories)
        noise_file = random.choice(self.files[category])
        
        # ------------ setting noise audio --------------- #
        n = torchaudio.load(noise_file)
        n = torch.squeeze(n)
        if len(x) > len(n):
            residual = len(x) - len(n)
            noise = [n]
            for i in range(0, residual // len(n)):
                noise.append(n)
            noise.append(n[0 : residual % len(n)])
            n = torch.cat(noise, dim = 0)
        elif len(x) < len(n):
            residual = len(n) - len(x)
            start = random.randint(0, residual-1)
            n = n[start : start + residual]
        
        # ------------ calculate decibels ---------------- #
        snr_db = torch.log10(snr)
        
        x_db = self.calculate_decibel(x)
        n_db = self.calculate_decibel(n)
        
        p = (x_db - snr_db - n_db)
        
        return x + (10 ** p)*n
    
    def calculate_decibel(self, x:torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): amplitude by time
        """
        m = torch.mean(x ** 2)
        m = torch.sqrt(m)
        return torch.log10(m + 1e-12)
        