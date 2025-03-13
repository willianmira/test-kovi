import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ChurnMetrics:
    def __init__(self, transactions_df):
        """Inicializa a classe com o DataFrame de transações"""
        if not isinstance(transactions_df, pd.DataFrame):
            raise ValueError("transactions_df deve ser um pandas DataFrame")
            
        required_columns = ['driver_id', 'amount']
        if not all(col in transactions_df.columns for col in required_columns):
            raise ValueError(f"DataFrame deve conter as colunas: {required_columns}")
            
        self.transactions = transactions_df.copy()
        
        # Renomear coluna date para transaction_date para manter consistência
        if 'date' in self.transactions.columns:
            self.transactions = self.transactions.rename(columns={'date': 'transaction_date'})
        elif 'transaction_date' not in self.transactions.columns:
            raise ValueError("DataFrame deve conter uma coluna 'date' ou 'transaction_date'")
            
        # Converter para datetime e ordenar
        self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])
        self.transactions = self.transactions.sort_values('transaction_date')
        
    def calculate_basic_metrics(self, window_days=30):
        """Calcula métricas básicas de churn"""
        try:
            metrics = {}
            
            # Agrupar por motorista e calcular última transação
            last_transactions = self.transactions.groupby('driver_id')['transaction_date'].max()
            
            # Data de referência (última data no dataset)
            reference_date = self.transactions['transaction_date'].max()
            
            # Identificar churned drivers (sem transações nos últimos X dias)
            churned_drivers = last_transactions[
                last_transactions < (reference_date - timedelta(days=window_days))
            ]
            
            # Calcular métricas básicas
            metrics['total_drivers'] = self.transactions['driver_id'].nunique()
            metrics['churned_drivers'] = len(churned_drivers)
            metrics['churn_rate'] = metrics['churned_drivers'] / metrics['total_drivers']
            
            # Calcular receita em risco
            # 1. Calcular receita média mensal por motorista
            monthly_revenue = (
                self.transactions.groupby('driver_id')['amount']
                .sum()
                .div(window_days/30)  # Normalizar para mês
            )
            
            # 2. Calcular receita média dos motoristas ativos
            active_drivers = last_transactions[
                last_transactions >= (reference_date - timedelta(days=window_days))
            ].index
            active_monthly_revenue = monthly_revenue[active_drivers].mean()
            
            # 3. Estimar receita em risco (receita média * motoristas em risco de churn)
            drivers_at_risk = len(active_drivers) * metrics['churn_rate']  # Estimativa baseada na taxa histórica
            metrics['revenue_at_risk'] = active_monthly_revenue * drivers_at_risk
            
            return metrics
        except Exception as e:
            print(f"Erro ao calcular métricas básicas: {str(e)}")
            return None
    
    def calculate_advanced_metrics(self):
        """Calcula métricas avançadas de churn"""
        try:
            metrics = {}
            
            # Tempo médio até o churn
            driver_lifetimes = self.calculate_driver_lifetimes()
            if driver_lifetimes is not None:
                metrics['avg_days_to_churn'] = driver_lifetimes.mean()
                metrics['median_days_to_churn'] = driver_lifetimes.median()
            
            # Valor do cliente antes do churn
            avg_revenue = self.calculate_revenue_before_churn()
            if avg_revenue is not None:
                metrics['avg_revenue_before_churn'] = avg_revenue
            
            # Churn por tipo de pagamento
            if 'kind' in self.transactions.columns:
                metrics['churn_by_payment_type'] = self.calculate_churn_by_segment('kind')
            
            # Sazonalidade do churn
            seasonality = self.calculate_churn_seasonality()
            if seasonality is not None:
                metrics['churn_seasonality'] = seasonality
            
            return metrics
        except Exception as e:
            print(f"Erro ao calcular métricas avançadas: {str(e)}")
            return None
    
    def calculate_driver_lifetimes(self):
        """Calcula o tempo de vida dos motoristas"""
        try:
            driver_first_transaction = self.transactions.groupby('driver_id')['transaction_date'].min()
            driver_last_transaction = self.transactions.groupby('driver_id')['transaction_date'].max()
            
            return (driver_last_transaction - driver_first_transaction).dt.days
        except Exception as e:
            print(f"Erro ao calcular tempo de vida dos motoristas: {str(e)}")
            return None
    
    def calculate_revenue_before_churn(self, pre_churn_window=30):
        """Calcula receita média antes do churn"""
        try:
            churned_drivers = self.identify_churned_drivers()
            if not churned_drivers.empty:
                revenue_before_churn = []
                for driver in churned_drivers:
                    last_date = self.transactions[
                        self.transactions['driver_id'] == driver
                    ]['transaction_date'].max()
                    
                    # Calcular receita no período antes do churn
                    revenue = self.transactions[
                        (self.transactions['driver_id'] == driver) &
                        (self.transactions['transaction_date'] >= last_date - timedelta(days=pre_churn_window))
                    ]['amount'].sum()
                    
                    revenue_before_churn.append(revenue)
                
                return np.mean(revenue_before_churn) if revenue_before_churn else 0
            return 0
        except Exception as e:
            print(f"Erro ao calcular receita antes do churn: {str(e)}")
            return None
    
    def calculate_churn_by_segment(self, segment_column):
        """Calcula taxa de churn por segmento"""
        try:
            if segment_column not in self.transactions.columns:
                raise ValueError(f"Coluna {segment_column} não encontrada no DataFrame")
                
            segments = self.transactions[segment_column].unique()
            churn_by_segment = {}
            
            for segment in segments:
                segment_drivers = self.transactions[
                    self.transactions[segment_column] == segment
                ]['driver_id'].unique()
                
                churned_drivers = self.identify_churned_drivers(
                    driver_subset=segment_drivers
                )
                
                churn_rate = len(churned_drivers) / len(segment_drivers) if len(segment_drivers) > 0 else 0
                churn_by_segment[segment] = churn_rate
                
            return churn_by_segment
        except Exception as e:
            print(f"Erro ao calcular churn por segmento: {str(e)}")
            return None
    
    def calculate_churn_seasonality(self):
        """Analisa sazonalidade do churn"""
        try:
            churned_drivers = self.identify_churned_drivers()
            if not churned_drivers.empty:
                # Pegar última data de cada motorista que deu churn
                churn_dates = self.transactions[
                    self.transactions['driver_id'].isin(churned_drivers)
                ].groupby('driver_id')['transaction_date'].max()
                
                # Agregar por mês
                monthly_churn = churn_dates.dt.to_period('M').value_counts().sort_index()
                
                return monthly_churn.to_dict()
            return {}
        except Exception as e:
            print(f"Erro ao calcular sazonalidade do churn: {str(e)}")
            return None
    
    def identify_churned_drivers(self, window_days=30, driver_subset=None):
        """Identifica motoristas que deram churn"""
        try:
            if driver_subset is None:
                transactions = self.transactions
            else:
                transactions = self.transactions[
                    self.transactions['driver_id'].isin(driver_subset)
                ]
            
            last_transactions = transactions.groupby('driver_id')['transaction_date'].max()
            reference_date = transactions['transaction_date'].max()
            
            churned_drivers = last_transactions[
                last_transactions < (reference_date - timedelta(days=window_days))
            ].index
            
            return churned_drivers
        except Exception as e:
            print(f"Erro ao identificar motoristas com churn: {str(e)}")
            return pd.Index([]) 