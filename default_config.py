
default_config = {
"model_name": "meta-llama/Llama-3.2-3B",
"experiment_name": "default-name",
"sentence_transformer_model": "BAAI/bge-m3",
"checkpoint_path": "pt_models",
"graph_path": "gnd/gnd.pickle",
"dataset_path": "dataset",
"label_mapping_path": "mapping/label_mapping.feather",
"num_epochs": 2,
"batch_size": 32,
"learning_rate": 5e-5,
"warmup_rate": 0.02,
"max_grad_norm": 1.0,
"eval_steps": 500,
"logging_steps": 50,
"save_steps": 5000,
"eval_generation": True,
"train_subsample_ratio": 1.0,
"validate_subsample_ratio": 0.03,
"sort_by_freq": True,
"use_k_freq_labels": 0,
"learning_rate_decay": True,

"context": {
  "context_type": "text", # Choices: "text" or "graph". Use null if no context is needed.
  "top_k": 0, # How many instances to retrieve
  "hops": 0, # Also include hops in the knowledge graph from retrieved instances
  "title_wise": False, # Whether to retrieve title-to-label or title-to-title
  "relation": None, # Choices: null (use all relations) "broader" or "related"
  "index_path": None,
  "mapping_path": None,
},

"prompt_config": {
    "num_prompt_tokens": 50,
    "at_layer": 25,
    "hidden_size": 3072,
    "down_project_size": 1024,
    "dropout": 0.1,
    "type": "hidden_states", # Choices: "hidden_states", "random", "context" or "graph_context"
    "gnn_hidden_size": 512,
    "gnn_n_layers": 2,
    "kge_size": 1024,
    "kge_kind": "random",
    "kge_path": None,
    "kge_encoder": None,
}
}