import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Carregar os dados
df = pd.read_csv("../data/transactions.csv")
df['date'] = pd.to_datetime(df['date'])

# Calcular métricas de churn
def calculate_churn_metrics(data):
    data_final = data['date'].max()
    df_driver = data.groupby('driver_id').agg(
        ultimo_pagamento=('date', 'max'),
        total_pagamentos=('date', 'count'),
        valor_medio=('amount', 'mean'),
        pagamentos_recorrentes=('kind', lambda x: (x == 'RECURRENCY').sum()),
        trocas_carro=('kind', lambda x: (x == 'FIRST_PAYMENT_EXCHANGE').sum()),
        contratos_novos=('kind', lambda x: (x == 'FIRST_PAYMENT').sum())
    ).reset_index()

    df_driver['dias_sem_pagar'] = (data_final - df_driver['ultimo_pagamento']).dt.days
    df_driver['churn'] = df_driver['dias_sem_pagar'] > 28
    return df_driver

df_driver = calculate_churn_metrics(df)

# Inicializar o app Dash
app = dash.Dash(__name__)

# Layout do dashboard
app.layout = html.Div([
    html.H1("Dashboard de Análise de Churn - Buddha Locadora", style={'textAlign': 'center'}),
    
    # Dropdown para filtrar por tipo de pagamento
    dcc.Dropdown(
        id='kind-dropdown',
        options=[{'label': kind, 'value': kind} for kind in df['kind'].unique()],
        value=df['kind'].unique()[0],
        multi=False,
        placeholder="Selecione o tipo de pagamento"
    ),
    
    # Gráfico de barras
    dcc.Graph(id='bar-chart'),
    
    # Gráfico de dispersão
    dcc.Graph(id='scatter-plot')
])

# Callback para atualizar os gráficos
@app.callback(
    [Output('bar-chart', 'figure'),
     Output('scatter-plot', 'figure')],
    [Input('kind-dropdown', 'value')]
)
def update_graphs(selected_kind):
    # Filtrar dados pelo tipo de pagamento selecionado
    filtered_df = df[df['kind'] == selected_kind]
    
    # Gráfico de barras: Total de pagamentos por motorista
    bar_chart = px.bar(
        filtered_df.groupby('driver_id').size().reset_index(name='total_pagamentos'),
        x='driver_id',
        y='total_pagamentos',
        title=f"Total de Pagamentos por Motorista ({selected_kind})"
    )
    
    # Gráfico de dispersão: Valor médio vs dias sem pagar
    scatter_plot = px.scatter(
        df_driver,
        x='valor_medio',
        y='dias_sem_pagar',
        color='churn',
        title="Valor Médio vs Dias Sem Pagar",
        labels={'valor_medio': 'Valor Médio', 'dias_sem_pagar': 'Dias Sem Pagar'}
    )
    
    return bar_chart, scatter_plot

# Salvar o dashboard como HTML
if __name__ == '__main__':
    app.run_server(debug=False)  # Executa o servidor localmente
    # Exportar para HTML
    with open("../results/insights_dashboard.html", "w") as f:
        f.write(app.index())