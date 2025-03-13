from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import pandas as pd
from src.ai_integration import FeatureEmbedder  # Importa a classe de embeddings

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para gerar embeddings de features.
    Usa a API do Hugging Face para processar características numéricas/categóricas.
    """
    def __init__(self, columns, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.columns = columns
        self.feature_embedder = FeatureEmbedder(model_name=model_name)

    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        """
        Gera embeddings para as features selecionadas.
        """
        return self.feature_embedder.embed_features(X, self.columns)


class ChurnModel:
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.best_model = None
    
    def train_model(self, model_type='rf'):
        """Treina o modelo de churn usando grid search"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV
        
        # Definir pipeline
        if model_type == 'rf':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=42))
            ])
            
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [5, 10]
            }
            
        else:  # gradient boosting
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(random_state=42))
            ])
            
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.1, 0.01]
            }
        
        # Realizar grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Guardar o melhor modelo
        self.best_model = grid_search
        
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        return grid_search
    
    def evaluate_model(self, model):
        """Avalia o modelo treinado"""
        from sklearn.metrics import roc_auc_score, classification_report
        import numpy as np
        
        # Fazer predições
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calcular métricas
        roc_auc = roc_auc_score(self.y_test, y_proba)
        print(f"AUC-ROC: {roc_auc}")
        
        # Imprimir relatório de classificação
        print("Relatório de Classificação:")
        print(classification_report(self.y_test, y_pred))
        
        # Retornar métricas para uso posterior
        return {
            'confusion_matrix': np.array([[sum((self.y_test == 0) & (y_pred == 0)), 
                                         sum((self.y_test == 0) & (y_pred == 1))],
                                        [sum((self.y_test == 1) & (y_pred == 0)), 
                                         sum((self.y_test == 1) & (y_pred == 1))]]),
            'roc_auc': roc_auc,
            'feature_names': self.feature_names,
            'feature_importance': model.best_estimator_['model'].feature_importances_,
            'y_prob': y_proba,
            'y_pred': y_pred,
            'y_true': self.y_test
        }
    
    def save_model(self, path):
        """Salva o modelo treinado"""
        import pickle
        import os
        
        if self.best_model is None:
            raise ValueError("Nenhum modelo treinado encontrado. Execute train_model primeiro.")
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Salvar o modelo
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"Modelo salvo com sucesso em {path}")
        except Exception as e:
            print(f"Erro ao salvar modelo: {str(e)}")
    
    def load_model(self, path):
        """Carrega um modelo salvo"""
        import pickle
        
        try:
            with open(path, 'rb') as f:
                self.best_model = pickle.load(f)
            print(f"Modelo carregado com sucesso de {path}")
            return self.best_model
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            return None

class DataPreparation:
    def __init__(self, data):
        self.data = data

    def prepare_features(self, target_column, features_to_drop=None):
        # Análise de missing values
        missing_analysis = self.data.isnull().sum()
        print("Análise de valores faltantes:")
        print(missing_analysis[missing_analysis > 0])
        
        # Tratamento de outliers usando IQR (Interquartile Range)
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if col != target_column:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Substituir outliers pela mediana
                median_value = self.data[col].median()
                self.data.loc[self.data[col] < lower_bound, col] = median_value
                self.data.loc[self.data[col] > upper_bound, col] = median_value
        
        # Feature engineering específico para churn
        if 'ultima_data_ativo' in self.data.columns:
            self.data['dias_inativo'] = (pd.Timestamp.now() - pd.to_datetime(self.data['ultima_data_ativo'])).dt.days
        
        if 'valor_medio_corrida' in self.data.columns and 'numero_corridas' in self.data.columns:
            self.data['receita_total'] = self.data['valor_medio_corrida'] * self.data['numero_corridas']
        
        if 'data_cadastro' in self.data.columns:
            self.data['tempo_cadastro'] = (pd.Timestamp.now() - pd.to_datetime(self.data['data_cadastro'])).dt.days
        
        # Preparar features finais
        if features_to_drop is None:
            features_to_drop = []
            
        # Remover colunas que não serão usadas no modelo
        features_to_drop.extend(['ultima_data_ativo', 'data_cadastro'])  # Removendo colunas de data brutas
        features_to_drop = list(set(features_to_drop))  # Removendo duplicatas
        
        # Separar features e target
        X = self.data.drop(columns=[target_column] + [col for col in features_to_drop if col in self.data.columns])
        y = self.data[target_column]
        
        print("\nFeatures selecionadas para o modelo:")
        print(X.columns.tolist())
        
        return X, y