"""
Adapted from SimCSE
"""


import torch
import torch.nn as nn

from functools import reduce


POOLER_TYPES = ['cls',
                'avg',
                'last_hidden',
                'last_second_hidden',
                'avg_all_hidden',
                'avg_second_to_last_hidden',
                'avg_first_last',
                'avg_last2',
                'avg_last4',
                'max_all_hidden',
                'max_second_to_last_hidden',
                'max_first_last',
                'max_last2',
                'max_last4',
                'concat_last4']


class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type

        if pooler_type not in POOLER_TYPES:
            raise NotImplementedError

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type == 'cls':
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "last_hidden":
            return last_hidden
        elif self.pooler_type == "last_second_hidden":
            second_last_hidden = hidden_states[-2]
            return second_last_hidden
        elif self.pooler_type == "avg_all_hidden":
            pooled_output = torch.stack(outputs.hidden_states).mean(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_second_to_last_hidden":
            pooled_output = torch.stack(outputs.hidden_states[1:]).mean(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_output = torch.stack([first_hidden, last_hidden]).mean(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_last2":
            pooled_output = torch.stack(outputs.hidden_states[-2:]).mean(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_last4":
            pooled_output = torch.stack(outputs.hidden_states[-4:]).mean(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "max_all_hidden":
            pooled_output, _ = torch.stack(outputs.hidden_states).max(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "max_second_to_last_hidden":
            pooled_output, _ = torch.stack(outputs.hidden_states[1:]).max(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "max_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_output, _ = torch.stack([first_hidden, last_hidden]).max(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "max_last2":
            pooled_output, _ = torch.stack(outputs.hidden_states[-2:]).max(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "max_last4":
            pooled_output, _ = torch.stack(outputs.hidden_states[-4:]).max(0)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "concat_last4":
            pooled_output = torch.cat(outputs.hidden_states[-4:], dim=2)
            pooled_result = (pooled_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError
