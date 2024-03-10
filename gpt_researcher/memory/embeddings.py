from llama_index.embeddings.ollama import OllamaEmbedding


class Memory:
    def __init__(self, **kwargs):
        self._embeddings = OllamaEmbedding(model_name='nomic-embed-text')

    def get_embeddings(self):
        return self._embeddings

