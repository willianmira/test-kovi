from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class FeatureEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, text):
        """Gera um embedding para um dado texto."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def embed_features(self, df, columns):
        """
        Gera embeddings para colunas numéricas/categóricas.
        Converte os valores em texto antes de gerar embeddings.
        """
        embeddings = []
        for _, row in df.iterrows():
            feature_text = " ".join([f"{col} {row[col]}" for col in columns])
            embedding = self.get_embedding(feature_text)
            embeddings.append(embedding.flatten())
        return np.array(embeddings)
    
    
    