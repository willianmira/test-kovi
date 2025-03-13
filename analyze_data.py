import pandas as pd
import numpy as np

# Ler os dados
df = pd.read_csv('data/processed/processed_transactions.csv')
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Análise mensal
monthly_stats = df.groupby(df['transaction_date'].dt.to_period('M')).agg({
    'driver_id': 'nunique',
    'amount': ['sum', 'mean', 'count']
}).round(2)

print('\nEstatísticas Mensais:')
print(monthly_stats)

# Análise de churn mensal
def calculate_monthly_churn(df):
    monthly_churn = []
    months = sorted(df['transaction_date'].dt.to_period('M').unique())
    
    for i in range(1, len(months)):
        current_month = months[i]
        prev_month = months[i-1]
        
        # Motoristas do mês anterior
        prev_drivers = set(df[df['transaction_date'].dt.to_period('M') == prev_month]['driver_id'])
        
        # Motoristas do mês atual
        current_drivers = set(df[df['transaction_date'].dt.to_period('M') == current_month]['driver_id'])
        
        # Motoristas que não continuaram
        churned = len(prev_drivers - current_drivers)
        
        # Taxa de churn
        churn_rate = churned / len(prev_drivers) if prev_drivers else 0
        
        monthly_churn.append({
            'month': current_month,
            'active_prev': len(prev_drivers),
            'active_current': len(current_drivers),
            'churned': churned,
            'churn_rate': churn_rate
        })
    
    return pd.DataFrame(monthly_churn)

churn_analysis = calculate_monthly_churn(df)
print('\nAnálise de Churn:')
print(churn_analysis)

# Análise por segmento
segment_analysis = df.groupby('kind').agg({
    'driver_id': 'nunique',
    'amount': ['sum', 'mean', 'count']
}).round(2)

print('\nAnálise por Segmento:')
print(segment_analysis) 