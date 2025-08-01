from copy import deepcopy
import torch
from torch import nn

class RandomPromptGenerator(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_prompt_tokens = config["num_prompt_tokens"]
        self.hidden_size = config["hidden_size"]
        self.prompt = nn.Embedding(
            self.num_prompt_tokens, self.hidden_size
        )
        self.proj_down = nn.Linear(config["hidden_size"], config["down_project_size"])
        self.intermediate_act_fn = nn.SiLU()   
        self.proj_up = nn.Linear(config["down_project_size"],  config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self):
        ids = torch.arange(self.num_prompt_tokens, device=self.prompt.weight.device)
        project_down = self.proj_down(self.prompt(ids))
        project_down = self.intermediate_act_fn(project_down)
        project_up = self.proj_up(project_down)
        apply_dropout = self.dropout(project_up)
        return apply_dropout

class HiddenStatePromptGenerator(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.num_prompt_tokens = config["num_prompt_tokens"]
        self.hidden_size = config["hidden_size"]
        self.proj_down = nn.Linear(config["hidden_size"], config["down_project_size"])
        self.intermediate_act_fn = nn.ReLU()   
        self.proj_up = nn.Linear(config["down_project_size"],  config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.num_prompt_tokens)
    
    def forward(self, hidden_states, seq_lengths):
        hidden_states = self.proj_down(hidden_states)
        batch_prompts = []
        for i in range(hidden_states.size(0)):
            hidden_state = hidden_states[i]
            hidden_state = hidden_state[0:seq_lengths[i], :].unsqueeze(0)
            hidden_state = hidden_state.transpose(1, 2) # B x D x L
            hidden_state = (self.adaptive_pooling(hidden_state)).transpose(1, 2) # B x num_prompt_tokens x D
            batch_prompts.append(hidden_state)

        hidden_states = torch.cat(batch_prompts, dim=0)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.proj_up(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states 

class ContextPromptGenerator(nn.Module):

    def __init__(self, config, embed) -> None:
        super().__init__()
        self.num_prompt_tokens = config["num_prompt_tokens"]
        self.embed = deepcopy(embed)
        self.hidden_size = config["hidden_size"]
        self.proj_down = nn.Linear(config["hidden_size"], config["down_project_size"])
        self.proj_down_context = nn.Linear(config["hidden_size"], config["down_project_size"])
        self.intermediate_act_fn = nn.SiLU()   
        self.proj_up = nn.Linear(config["down_project_size"],  config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.num_prompt_tokens)
    
    def forward(self, context_ids, hidden_states, seq_lengths, context_lengths):
        context_hidden_states = self.embed(context_ids)
        mean_context_hidden_states = []
        for i in range(context_hidden_states.size(0)):
            context_hidden_i = context_hidden_states[i]
            context_hidden_i = context_hidden_i[0:context_lengths[i], :].unsqueeze(0)
            mean_context_hidden_states.append(context_hidden_i.mean(dim=1))
        context_hidden_states = torch.cat(mean_context_hidden_states, dim=0)
        context_hidden_states = self.proj_down_context(context_hidden_states)
        hidden_states = self.proj_down(hidden_states)  
        batch_prompts = []
        for i in range(hidden_states.size(0)):
            context_hidden_i = context_hidden_states[i]
            context_hidden_i = context_hidden_i.unsqueeze(0).unsqueeze(0)
            hidden_state = hidden_states[i]
            hidden_state = hidden_state[0:seq_lengths[i], :].unsqueeze(0)
            hidden_state = torch.cat([context_hidden_i, hidden_state], dim=1) 
            hidden_state = hidden_state.transpose(1, 2) # B x D x L
            hidden_state = (self.adaptive_pooling(hidden_state)).transpose(1, 2) # B x num_prompt_tokens x D
            batch_prompts.append(hidden_state)

        hidden_states = torch.cat(batch_prompts, dim=0)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.proj_up(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states 
