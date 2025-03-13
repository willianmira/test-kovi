import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

class DataPreparation:
    def __init__(self, data_source):
        """
        Inicializa a classe DataPreparation
        
        Args:
            data_source: Pode ser um DataFrame do pandas ou uma string com o caminho do arquivo
        """
        if isinstance(data_source, str):
            self.data = self.load_data(data_source)
        else:
            self.data = data_source.copy()
        
        if self.data is not None:
            self._create_features()
    
    def load_data(self, file_path):
        """Carrega os dados do arquivo CSV"""
        try:
            print(f"Carregando dados de {file_path}")
            df = pd.read_csv(file_path)
            print(f"Dados carregados com sucesso. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Erro ao carregar dados: {str(e)}")
            return None
    
    def _create_features(self):
        """Cria features básicas a partir dos dados brutos"""
        # Converter data para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Features por motorista
        driver_features = self.data.groupby('driver_id').agg({
            'amount': ['count', 'mean', 'std', 'sum'],
            'date': ['min', 'max']
        }).reset_index()
        
        # Renomear colunas
        driver_features.columns = [
            'driver_id',
            'total_transactions',
            'avg_amount',
            'std_amount',
            'total_amount',
            'first_transaction',
            'last_transaction'
        ]
        
        # Calcular features temporais
        driver_features['days_active'] = (
            driver_features['last_transaction'] - driver_features['first_transaction']
        ).dt.days
        
        driver_features['avg_transaction_per_day'] = (
            driver_features['total_transactions'] / 
            (driver_features['days_active'] + 1)  # +1 para evitar divisão por zero
        )
        
        # Calcular churn (30 dias sem transação)
        reference_date = self.data['date'].max()
        driver_features['churn'] = (
            reference_date - driver_features['last_transaction']
        ).dt.days > 30
        
        # Converter para int
        driver_features['churn'] = driver_features['churn'].astype(int)
        
        # Features de tipo de pagamento
        payment_types = pd.get_dummies(self.data['kind'], prefix='payment_type')
        payment_features = self.data.join(payment_types).groupby('driver_id')[
            payment_types.columns
        ].mean()
        
        # Juntar todas as features
        self.features = driver_features.merge(
            payment_features,
            on='driver_id',
            how='left'
        )
    
    def prepare_features(self, target_column='churn', test_size=0.2, random_state=42):
        """Prepara features para modelagem"""
        try:
            # Verificar se temos as features necessárias
            if not hasattr(self, 'features'):
                self._create_features()
            
            # Separar features e target
            X = self.features.drop(columns=['driver_id', target_column, 'first_transaction', 'last_transaction'])
            y = self.features[target_column]
            
            # Guardar nomes das colunas
            feature_columns = X.columns.tolist()
            
            # Escalonar features
            scaler = StandardScaler()
            X = pd.DataFrame(
                scaler.fit_transform(X),
                columns=feature_columns
            )
            
            # Split treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            
            return X_train, X_test, y_train, y_test, feature_columns
            
        except Exception as e:
            print(f"Erro ao preparar features: {str(e)}")
            return None, None, None, None, None
    
    def get_feature_descriptions(self):
        """Retorna descrição das features criadas"""
        return {
            'total_transactions': 'Número total de transações do motorista',
            'avg_amount': 'Valor médio das transações',
            'std_amount': 'Desvio padrão dos valores das transações',
            'total_amount': 'Valor total transacionado',
            'days_active': 'Número de dias entre primeira e última transação',
            'avg_transaction_per_day': 'Média de transações por dia',
            'payment_type_FIRST_PAYMENT': 'Proporção de primeiros pagamentos',
            'payment_type_FIRST_PAYMENT_EXCHANGE': 'Proporção de trocas no primeiro pagamento',
            'payment_type_RECURRENCY': 'Proporção de pagamentos recorrentes'
        } 