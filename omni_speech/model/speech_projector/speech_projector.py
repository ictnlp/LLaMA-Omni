# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/projector.py


import torch
import torch.nn as nn


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.hidden_size)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x