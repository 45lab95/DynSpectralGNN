# models/sequence_encoder.py

import torch
import torch.nn as nn
from typing import Optional, Tuple

class LSTMWrapper(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 batch_first: bool = True,
                 dropout: float = 0.0,
                 bidirectional: bool = False):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first  
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0, 
            bidirectional=bidirectional
        )

    def forward(self,
                sequence_input: torch.Tensor,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        output_sequence, final_state = self.lstm(sequence_input, initial_state)

        return output_sequence, final_state

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)