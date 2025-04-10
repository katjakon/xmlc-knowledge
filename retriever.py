import faiss
from sentence_transformers import SentenceTransformer
from utils import k_hop_neighbors

class Retriever:

    def __init__(self, retriever_model, M=200, device=None) -> None:
        self.retriever = SentenceTransformer(retriever_model)
        if device is not None:
            self.retriever.to(device)
        self.dim = self.retriever.get_sentence_embedding_dimension()
        self.M = M
    
    def retrieve(self, mapping, labels, texts, top_k=10, batch_size=256, index=None):
        """
        labels: list of strings
        mapping: dict
        texts: list of strings
        top_k: int
        batch_size: int

        Returns:
        similarity: np.ndarray of shape (len(texts), top_k)
        indices: np.ndarray of shape (len(texts), top_k)
        """
        if index is None:
            index = faiss.IndexHNSWFlat(self.dim, self.M)
            label_embeddings = self.retriever.encode(labels, show_progress_bar=True, batch_size=batch_size)
            index.add(label_embeddings)

        text_embeddings = self.retriever.encode(texts, show_progress_bar=False, batch_size=batch_size)
        similarity, indices = index.search(text_embeddings, top_k)
        label_idn = [list(map(lambda idx: mapping[idx], top_indices)) for top_indices in indices]
        return similarity, label_idn
    
    def fit(self, labels, batch_size=256):
        index = faiss.IndexHNSWFlat(self.dim, self.M)
        label_embeddings = self.retriever.encode(labels, show_progress_bar=True, batch_size=batch_size)
        index.add(label_embeddings)
        return index
    
    def get_neighbors(self, list_idns, graph, k, relation=None):
        retrieved_labels_plus = []
        for top_labels in list_idns:
            extended_labels = set()
            for label_idn in top_labels:
                extended_labels.add(label_idn)
                neighbors = k_hop_neighbors(graph, label_idn, k, relation)
                extended_labels.update(neighbors)
            retrieved_labels_plus.append(list(extended_labels))
        return retrieved_labels_plus
    
    def retrieve_with_neighbors(self, graph, mapping, labels, texts, k=2, top_k=10, batch_size=256, relation=None):
        """
        graph: networkx.Graph
        mapping: dict
        labels: list of strings
        texts: list of strings
        k: int
        top_k: int
        batch_size: int

        Returns:
        retrieved_labels_plus: list of list of strings
        """
        _, idns = self._retrieve(labels, texts, top_k, batch_size)
        retrieved_labels_plus = self.get_neighbors(idns, graph, mapping, k, relation)
        return retrieved_labels_plus
                