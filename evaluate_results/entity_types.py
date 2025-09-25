
def add_entity_info(df, graph):
    df["entity"] = df["label-id"].apply(
    lambda x: graph.node_type(x) 
    )
    return df