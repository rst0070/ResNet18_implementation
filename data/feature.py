import torch
import torch.nn.functional as F
import torchaudio.transforms as ts
import torch.nn as nn


class AudioPreEmphasis(nn.Module):

    def __init__(self, coeff=0.97):
        super(AudioPreEmphasis, self).__init__()

        self.w = torch.FloatTensor([-coeff, 1.0]).unsqueeze(0).unsqueeze(0)

    def forward(self, audio):
        audio = F.pad(audio,(1,0), 'reflect')
        return F.conv1d(audio, self.w.to(audio.device))
    
class LogMelSpec(nn.Module):

    def __init__(self, exp_args):
        """
        Args
            exp_args - exp args from arguments.py
        """
        super(LogMelSpec, self).__init__()
        self.melspec = ts.MelSpectrogram(
            sample_rate = exp_args['sample_rate'], 
            n_fft = exp_args['n_fft'], 
            n_mels = exp_args['n_mels'], 
            win_length = exp_args['win_length'], 
            hop_length = exp_args['hop_length'], 
            window_fn=torch.hamming_window
        )
        
    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): waveform
        return:
            Log mel spectrogram(2-dimensional)
        """
        x = self.melspec(x)
        x = torch.log(x + 1e-12)
        return x
