import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineering(BaseEstimator, TransformerMixin):
    """Classe para criação de features avançadas para análise de churn"""
    
    def __init__(self):
        self.feature_names_ = None
        self.temporal_features = None
        self.behavioral_features = None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transforma o dataset criando novas features"""
        df = X.copy()
        
        # Renomear coluna date para transaction_date para manter consistência
        df = df.rename(columns={'date': 'transaction_date'})
        
        # 1. Features Temporais
        df = self._create_temporal_features(df)
        
        # 2. Features Comportamentais
        df = self._create_behavioral_features(df)
        
        # 3. Features de Valor
        df = self._create_monetary_features(df)
        
        # 4. Features de Risco
        df = self._create_risk_features(df)
        
        # 5. Features de Interação
        df = self._create_interaction_features(df)
        
        return df
    
    def _create_temporal_features(self, df):
        """Cria features temporais"""
        # Converter para datetime
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Features de tempo
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        df['month'] = df['transaction_date'].dt.month
        df['quarter'] = df['transaction_date'].dt.quarter
        df['is_weekend'] = df['transaction_date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Frequência de transações
        transaction_counts = df.groupby('driver_id')['transaction_date'].agg(['count', 'min', 'max'])
        df['transaction_frequency'] = df['driver_id'].map(transaction_counts['count'])
        
        # Tempo desde primeira/última transação
        df['days_since_first'] = (df['transaction_date'] - df['driver_id'].map(transaction_counts['min'])).dt.days
        df['days_since_last'] = (df['transaction_date'] - df['driver_id'].map(transaction_counts['max'])).dt.days
        
        return df
    
    def _create_behavioral_features(self, df):
        """Cria features comportamentais"""
        # Padrões de pagamento
        payment_patterns = df.groupby('driver_id')['amount'].agg(['mean', 'std', 'min', 'max'])
        df['avg_payment'] = df['driver_id'].map(payment_patterns['mean'])
        df['payment_volatility'] = df['driver_id'].map(payment_patterns['std'])
        df['payment_range'] = df['driver_id'].map(payment_patterns['max'] - payment_patterns['min'])
        
        # Consistência de pagamentos
        df['payment_consistency'] = df['amount'] / df['avg_payment']
        
        # Padrões de uso
        df['usage_intensity'] = df.groupby('driver_id')['amount'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        return df
    
    def _create_monetary_features(self, df):
        """Cria features monetárias"""
        # Valor total por motorista
        df['total_value'] = df.groupby('driver_id')['amount'].transform('sum')
        
        # Valor médio por período
        df['avg_value_per_month'] = df.groupby(['driver_id', df['transaction_date'].dt.to_period('M')])['amount'].transform('mean')
        
        # Tendência de valor
        df['value_trend'] = df.groupby('driver_id')['amount'].transform(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        return df
    
    def _create_risk_features(self, df):
        """Cria features de risco"""
        # Indicadores de risco baseados em padrões
        df['late_payment_risk'] = (
            (df['amount'] < df['avg_payment']) &
            (df['payment_consistency'] < 0.8)
        ).astype(int)
        
        # Score de risco composto
        risk_factors = [
            df['payment_volatility'] / df['avg_payment'],
            df['late_payment_risk'],
            -df['transaction_frequency']  # Menos transações = maior risco
        ]
        df['risk_score'] = sum(risk_factors) / len(risk_factors)
        
        return df
    
    def _create_interaction_features(self, df):
        """Cria features de interação"""
        # Interações entre features numéricas
        df['value_frequency_interaction'] = df['avg_payment'] * df['transaction_frequency']
        df['risk_value_interaction'] = df['risk_score'] * df['total_value']
        
        # Interações temporais
        df['seasonal_value'] = df['amount'] * np.sin(2 * np.pi * df['month'] / 12)
        
        return df
    
    def get_feature_names(self):
        """Retorna nomes das features criadas"""
        return self.feature_names_