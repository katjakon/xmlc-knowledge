import networkx as nx

from typing import List, Dict, Any


class GNDGraph(nx.DiGraph):

    def __init__(self, incoming_graph_data=None) -> None:
        """
        Initializes the GNDGraph with a NetworkX graph.
        Args:
            nx_graph (networkx.Graph): A NetworkX graph representing the GND.
        """
        super().__init__(incoming_graph_data)
        self.incoming_graph_data = incoming_graph_data
    
    def pref_label_name(self, node_id: str) -> str:
        """
        Returns the preferred label name for a given node ID.
        Args:
            node_id (str): The ID of the node.
        Returns:
            str: The preferred label name. None if the node does not exist or has no preferred name.
        """
        node_data = self.nodes.get(node_id)
        if node_data is None:
            return None
        prefered_name = node_data.get("pref", None)
        if prefered_name is not None:
            return list(prefered_name)[0]
        else:
            return None
    
    def alt_label_names(self, node_id: str) -> List[str]:
        """
        Returns alternative label names for a given node ID.
        Args:
            node_id (str): The ID of the node.
        Returns:
            List[str]: A list of alternative label names. Empty list if the node does not exist or has no alternative names.
        """
        node_data = self.nodes.get(node_id)
        if node_data is None:
            return []
        alternative_names = node_data.get("alt", set())
        hidden_names = node_data.get("hidden", set())
        alternative_names.update(hidden_names)
        return list(alternative_names)

    
    def relation_type(self, head, tail):
        """
        Returns the relation type between two nodes.
        Args:
            head (str): The ID of the head node.
            tail (str): The ID of the tail node.
        Returns:
            str: The relation type. None if no relation exists.
        """
        edge_data = self.edges.get((head, tail))
        if edge_data is not None:
            return edge_data.get("relation")
        else:
            return None
    
    def node_type(self, node_id: str) -> str:
        """
        Returns the type of a node.
        Args:
            node_id (str): The ID of the node.
        Returns:
            str: The type of the node. None if the node does not exist or has no type.
        """
        node_data = self.nodes.get(node_id)
        if node_data is not None:
            return node_data.get("type")
        else:
            return None
        
    def neighborhood(self, node_id, k=1, relation=None):
        """
        Returns the k-hop neighborhood of a node.
        Args:
            node_id (str): The ID of the node.
            k (int): The number of hops to consider in the neighborhood.
            relation (str, optional): The type of relation to filter neighbors. Defaults to None.
        Returns:
            GNDGraph: A subgraph containing the k-hop neighborhood of the node.
        """
        neighbors = set()
        if k == 0 or node_id not in self.nodes:
            return neighbors
        for neighbor in self.neighbors(node_id):
            if relation is not None:
                rel_type = self.relation_type(node_id, neighbor)
                if rel_type != relation:
                    continue
            neighbors.add(neighbor)
            neighbors.update(self.neighborhood(neighbor, k-1, relation))
        return self.subgraph(neighbors)
    
    def mapping(self, with_alt_labels=False):
        """
        Return indices mapping for the nodes in the graph.
        Returns:
            Tuple[List[str], Dict[int, str]]: A tuple containing a list of label names and a mapping from indices to node IDs.
        """
        label_names = []
        mapping = {}
        for idx, node_id in enumerate(self.nodes()):
            label_description = self.pref_label_name(node_id)
            if with_alt_labels:
                alt_names = self.alt_label_names(node_id)
                alt_names_str = " ".join(alt_names)
                label_description = f"{label_description} {alt_names_str}"
            label_names.append(label_description)
            mapping[idx] = node_id
        return label_names, mapping
