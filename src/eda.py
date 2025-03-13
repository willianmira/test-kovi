import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, data):
        self.data = data

    def calculate_metrics(self):
        """Calcula métricas relevantes para análise de churn."""
        df_driver = self.data.groupby('driver_id').agg(
            ultimo_pagamento=('date', 'max'),
            primeiro_pagamento=('date', 'min'),
            total_pagamentos=('date', 'count'),
            valor_medio=('amount', 'mean'),
            valor_total=('amount', 'sum'),
            pagamentos_recorrentes=('kind', lambda x: (x == 'RECURRENCY').sum()),
            trocas_carro=('kind', lambda x: (x == 'FIRST_PAYMENT_EXCHANGE').sum()),
            contratos_novos=('kind', lambda x: (x == 'FIRST_PAYMENT').sum())
        ).reset_index()

        data_final = self.data['date'].max()
        df_driver['dias_sem_pagar'] = (data_final - df_driver['ultimo_pagamento']).dt.days
        df_driver['churn'] = df_driver['dias_sem_pagar'] > 28

        return df_driver

    def visualize_patterns(self, df_driver):
        """Gera visualizações para explorar padrões no churn."""
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df_driver, x='valor_medio', hue='churn', multiple='stack', bins=30)
        plt.title("Distribuição de Valor Médio por Churn")
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.barplot(x='contratos_novos', y='churn', data=df_driver)
        plt.title("Churn por Número de Contratos Novos")
        plt.show()

        # Heatmap de correlação
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_driver.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlação entre Variáveis")
        plt.show()

        # Gráfico de Sankey (inovador)
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=["Novo Contrato", "Pagamento Recorrente", "Churn"]),
            link=dict(source=[0, 1], target=[1, 2], value=[df_driver['contratos_novos'].sum(), df_driver['churn'].sum()])
        )])
        fig.update_layout(title_text="Fluxo de Clientes entre Tipos de Pagamento", font_size=10)
        fig.show()