# gpt.py
以下是一个简单的 GPT 模型代码   注意：这是一个基本版本的 GPT 模型，可能需要根据具体问题进行修改和优化。
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2Model.from_pretrained('gpt2')

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
model = GPT()
