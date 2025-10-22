from copy import deepcopy

import torch
from torch import nn
import torch_geometric as pyg

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


class GraphContextPromptGenerator(nn.Module):

    def __init__(self, config, embeddings) -> None:
        super().__init__()
        self.num_prompt_tokens = config["num_prompt_tokens"]
        self.hidden_size = config["hidden_size"]
        self.down_project_size =  config["down_project_size"]
        self.kge_size = config["kge_size"]
        self.gnn_hidden_size = config["gnn_hidden_size"]
        self.gnn_n_layers = config["gnn_n_layers"]
        self.proj_down = nn.Linear(self.hidden_size, self.down_project_size)
        self.proj_down_context = nn.Linear(self.kge_size, self.down_project_size)
        self.intermediate_act_fn = nn.SiLU()   
        self.proj_up = nn.Linear(self.down_project_size,  self.hidden_size)
        self.dropout = nn.Dropout(config["dropout"])
        self.embeddings = deepcopy(embeddings)
        self.gat = pyg.nn.GAT(
            in_channels=self.kge_size,
            out_channels=self.kge_size,
            hidden_channels=self.gnn_hidden_size ,
            num_layers=self.gnn_n_layers)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.num_prompt_tokens)
    
    def forward(self, graph_batch, hidden_states, seq_lengths):
        graph_x, graph_edge_index = graph_batch.x, graph_batch.edge_index
        graph_x = self.embeddings(graph_x)
        # Employ GNN on graph data.
        gat_out = self.gat(x=graph_x, edge_index=graph_edge_index)
        # Seperate individual graphs for each instance in batch.
        _, counts = torch.unique(graph_batch.batch, return_counts=True)
        original_graphs_out = torch.split(gat_out, counts.tolist()) # List with len=B with each item: num_nodes x kge size
        graph_context = []
        for graph_out in original_graphs_out:
            graph_out = self.proj_down_context(graph_out)
            graph_context.append(graph_out)
        hidden_states = self.proj_down(hidden_states)  
        batch_prompts = []
        for i in range(hidden_states.size(0)):
            context_hidden_i = graph_context[i]
            context_hidden_i = context_hidden_i.unsqueeze(0)
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

class FusedPromptGenerator(nn.Module):
    def __init__(self, config, freeze_prompt=True) -> None:
        super().__init__()
        self.num_prompt_tokens = config["num_prompt_tokens"]
        self.hidden_size = config["hidden_size"]
        self.prompt = nn.Embedding(
            self.num_prompt_tokens, self.hidden_size,
        )
        if freeze_prompt:
            self.prompt.weight.requires_grad = False
        self.proj_down = nn.Linear(config["hidden_size"], config["down_project_size"])
        self.proj_down_prompt = nn.Linear(config["hidden_size"], config["down_project_size"])
        self.intermediate_act_fn = nn.SiLU()   
        self.proj_up = nn.Linear(config["down_project_size"],  config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.num_prompt_tokens)

    def forward(self, hidden_states, seq_lengths):
        prompt = self.prompt.weight
        prompt = self.proj_down_prompt(prompt).unsqueeze(0)  # 1 x num_prompt_tokens x down_project_size
        hidden_states = self.proj_down(hidden_states)  
        batch_prompts = []
        for i in range(hidden_states.size(0)):
            hidden_state = hidden_states[i]
            hidden_state = hidden_state[0:seq_lengths[i], :].unsqueeze(0)
            hidden_state = torch.cat([prompt, hidden_state], dim=1) 
            hidden_state = hidden_state.transpose(1, 2) # B x D x L
            hidden_state = (self.adaptive_pooling(hidden_state)).transpose(1, 2) # B x num_prompt_tokens x D
            batch_prompts.append(hidden_state)

        hidden_states = torch.cat(batch_prompts, dim=0)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.proj_up(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states 