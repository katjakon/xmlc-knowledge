import faiss
import pickle
from sentence_transformers import SentenceTransformer

class Retriever:

    def __init__(self, retriever_model, graph, M=200, device=None) -> None:
        self.retriever = SentenceTransformer(retriever_model, device=device)
        self.dim = self.retriever.get_sentence_embedding_dimension()
        self.M = M
        self.index = None
        self.mapping = None
        self.graph = graph
    
    def retrieve(self, texts, top_k=10, batch_size=256):
        """
        texts: list of strings
        top_k: int
        batch_size: int

        Returns:
        similarity: np.ndarray of shape (len(texts), top_k)
        indices: np.ndarray of shape (len(texts), top_k)
        """
        if self.index is None or self.mapping is None:
            raise ValueError("Index  or Mapping have not been created or loaded yet.")
        text_embeddings = self.retriever.encode(texts, show_progress_bar=False, batch_size=batch_size)
        distance, indices = self.index.search(text_embeddings, top_k)
        # if remove_diagonal:
        #     for i in range(len(texts)):
        #         filter_indices = [idx for idx in indices[i] if idx != i]
        #         indices[i] = filter_indices
        # if title_wise: # Title retrieval returns lists of identifiers per index.
        #     label_idn = [[idn for label_idn_list in map(lambda idx: self.mapping[idx], top_indices) for idn in label_idn_list] for top_indices in indices]
        # else: # Label retrieval returns only one identifier per index.
        label_idn = [list(map(lambda idx: self.mapping[idx], top_indices)) for top_indices in indices]
        return distance, label_idn
    
    def fit(self, batch_size=256, with_alt_labels=False):
        label_strings, mapping = self.graph.mapping(with_alt_labels=with_alt_labels)
        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        embeddings = self.retriever.encode(label_strings, show_progress_bar=True, batch_size=batch_size)
        self.index.add(embeddings)
        self.mapping = mapping
        return self.index
    
    def embeddings(self, batch_size=256, with_alt_labels=False):
        label_strings, idx2idn = self.graph.mapping(with_alt_labels=with_alt_labels)
        embeddings = self.retriever.encode(label_strings, show_progress_bar=True, batch_size=batch_size)
        return idx2idn, embeddings
    
    def get_neighbors(self, list_idns, k, relation=None):
        retrieved_labels_plus = []
        for top_labels in list_idns:
            extended_labels = set()
            for label_idn in top_labels:
                extended_labels.add(label_idn)
                neighbors = self.graph.neighborhood(label_idn, k=k, relation=relation)
                neighbors = list(neighbors)
                extended_labels.update(neighbors)
            retrieved_labels_plus.append(list(extended_labels))
        return retrieved_labels_plus
    
    def retrieve_with_neighbors(self, texts, k=2, top_k=10, batch_size=256, relation=None):
        """
        texts: list of strings
        k: int, of the retrieved labels also get neighbors in k hops.
        top_k: int, retrieve top k similar labels
        batch_size: int

        Returns:
        retrieved_labels_plus: list of list of strings
        """
        _, idns = self.retrieve(
            texts=texts,
            top_k=top_k,
            batch_size=batch_size)
        retrieved_labels_plus = self.get_neighbors(idns, k, relation)
        return retrieved_labels_plus
    
    def load_search_index(self, index_path, mapping_path):
        """
        Loads a search index from a file.
        
        Args:
            path (str): Path to the index file.
        
        Returns:
            faiss.IndexHNSWFlat: Loaded index.

        """
        self.index = pickle.load(open(index_path, "rb"))
        self.mapping = pickle.load(open(mapping_path, "rb"))
        return self.index
    
    def save_search_index(self, index_path, mapping_path):
        """
        Saves the search index to a file.
        
        Args:
            path (str): Path to save the index file.
        """
        if self.index is None:
            raise ValueError("Index has not been created or loaded yet.")
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(mapping_path, "wb") as f:
            pickle.dump(self.mapping, f)
    
    
                