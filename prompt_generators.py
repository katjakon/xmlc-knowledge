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
        self.intermediate_act_fn = nn.SiLU()   
        self.proj_up = nn.Linear(config["down_project_size"],  self.num_prompt_tokens * config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])
    
    def forward(self, hidden_states):
        hidden_states = hidden_states.sum(dim=1)
        hidden_states = self.proj_down(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.proj_up(hidden_states)
        hidden_states = hidden_states.view(hidden_states.size(0), self.num_prompt_tokens, -1)
        hidden_states = self.dropout(hidden_states)
        return hidden_states 

