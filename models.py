import config
from transformers import RobertaModel, RobertaConfig
import torch.nn as nn
import torch

class TokenModel(nn.Module):
    def __init__(self):
        super(TokenModel, self).__init__()
        self.config = RobertaConfig.from_pretrained(config.roberta_config, output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained(config.roberta_model, config=self.config)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

  
    def forward(self, input_ids, attention_masks):
        _, _, hs = self.roberta(input_ids, attention_masks)
        hid_out = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
        hid_out = torch.mean(hid_out, 0)
        x = self.dropout(hid_out)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
            
        return start_logits, end_logits



class CharacterModel(nn.Module):
    def __init__(self):
        super(CharacterModel, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=config.lstm_hidden_size, num_layers=config.lstm_num_layers, bias=True, batch_first=True, dropout=0.2, bidirectional=True)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(config.lstm_hidden_size*config.lstm_direction, 2)
        self.tanh = nn.Tanh()
  
  

    def forward(self, input, h_0, c_0):
        x, (hn, cn) = self.lstm(input, (h_0, c_0))
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits