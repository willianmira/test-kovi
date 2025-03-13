import pandas as pd

# Ler os dados
df = pd.read_csv('data/processed/processed_transactions.csv')
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Função para calcular churn entre dois meses
def calculate_churn(df, current_month, previous_month=None):
    current_drivers = set(df[df['transaction_date'].dt.month == current_month]['driver_id'])
    
    if previous_month:
        previous_drivers = set(df[df['transaction_date'].dt.month == previous_month]['driver_id'])
        churned = previous_drivers - current_drivers
        churn_rate = len(churned) / len(previous_drivers) if previous_drivers else 0
        new_drivers = current_drivers - previous_drivers
    else:
        churned = set()
        churn_rate = 0
        new_drivers = current_drivers
        previous_drivers = set()
    
    return {
        'month': current_month,
        'active_drivers': len(current_drivers),
        'churned_drivers': len(churned),
        'new_drivers': len(new_drivers),
        'churn_rate': churn_rate,
        'previous_active': len(previous_drivers)
    }

# Analisar todos os meses
months = sorted(df['transaction_date'].dt.month.unique())
results = []

for i, month in enumerate(months):
    previous_month = months[i-1] if i > 0 else None
    result = calculate_churn(df, month, previous_month)
    results.append(result)

# Criar DataFrame com resultados
results_df = pd.DataFrame(results)
pd.set_option('display.float_format', '{:.2%}'.format)

# Imprimir resultados detalhados
print("\nAnálise detalhada mês a mês:")
print("=" * 100)
for result in results:
    month_name = pd.Timestamp(2024, result['month'], 1).strftime('%B')
    print(f"\nMês: {month_name}")
    print(f"Motoristas ativos: {result['active_drivers']}")
    print(f"Motoristas que deixaram a plataforma: {result['churned_drivers']}")
    print(f"Novos motoristas: {result['new_drivers']}")
    print(f"Taxa de evasão: {result['churn_rate']:.1%}")
    if result['previous_active'] > 0:
        print(f"Crescimento líquido: {((result['active_drivers'] - result['previous_active'])/result['previous_active']):.1%}")
    print("-" * 50)

# Análise adicional das transações
print("\nAnálise de transações por mês:")
monthly_stats = df.groupby(df['transaction_date'].dt.month).agg({
    'driver_id': 'nunique',
    'amount': ['count', 'sum', 'mean']
}).round(2)

print("\nEstatísticas mensais:")
print(monthly_stats)

# Verificar se existem dados de dezembro/2023
print("\nVerificação de período dos dados:")
print(f"Data mais antiga: {df['transaction_date'].min()}")
print(f"Data mais recente: {df['transaction_date'].max()}") 