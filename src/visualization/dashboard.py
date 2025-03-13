import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ChurnDashboard:
    def __init__(self, data, churn_metrics, model_metrics=None):
        self.data = data
        self.churn_metrics = churn_metrics
        self.model_metrics = model_metrics
        
        # Configurações de estilo
        self._setup_style_configs()
        
        # Inicialização do app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
    
    def _setup_style_configs(self):
        """Configura estilos e cores do dashboard."""
        # Cores
        self.colors = {
            'primary': '#2E5BFF',   # Azul
            'success': '#00C48C',   # Verde
            'warning': '#FFB946',   # Laranja
            'danger': '#FF4B4B',    # Vermelho
            'info': '#885AF8',      # Roxo
            'gray': '#E1E5ED',      # Cinza
            'text': '#3C4257',      # Texto
            'background': '#F7FAFC'  # Fundo
        }
        
        # Estilo base para gráficos
        self.plot_style = {
            'font_family': '"Inter", sans-serif',
            'title_font_size': 24,
            'title_font_family': '"Inter", sans-serif',
            'title_font_color': '#1A1F36',
            'title_x': 0.5,
            'title_xanchor': 'center',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'xaxis': {
                'gridcolor': '#E1E5ED',
                'linecolor': '#E1E5ED',
                'title_font': {'size': 14, 'color': '#3C4257'},
                'tickfont': {'size': 12, 'color': '#697386'}
            },
            'yaxis': {
                'gridcolor': '#E1E5ED',
                'linecolor': '#E1E5ED',
                'title_font': {'size': 14, 'color': '#3C4257'},
                'tickfont': {'size': 12, 'color': '#697386'}
            },
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1,
                'font': {'size': 12, 'color': '#3C4257'}
            },
            'margin': {'t': 80, 'b': 40, 'l': 40, 'r': 40}
        }
        
        # Estilo para cards
        self.card_style = {
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'padding': '24px',
            'marginBottom': '32px'
        }
        
        # Estilo para títulos
        self.title_style = {
            'fontFamily': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            'fontWeight': '600',
            'color': '#1A1F36',
            'marginBottom': '24px'
        }
        
        # Estilo para subtítulos
        self.subtitle_style = {
            'fontFamily': '"Inter", sans-serif',
            'fontWeight': '500',
            'color': '#3C4257',
            'marginBottom': '16px',
            'fontSize': '20px'
        }
    
    def setup_layout(self):
        """Define o layout principal do dashboard."""
        self.app.layout = html.Div([
            # Header
            self._create_header(),
            
            # Seções principais 
            self._create_trends_section(),
            self._create_segments_section(),
            self._create_risk_section(),
            self._create_strategic_recommendations_section()
            
        ], style={
            'padding': '24px 48px',
            'backgroundColor': self.colors['background'],
            'minHeight': '100vh',
            'fontFamily': '"Inter", sans-serif'
        })
        
        self._setup_callbacks()
    
    def _create_header(self):
        """Cria o cabeçalho do dashboard."""
        return html.Div([
            html.H1(
                "Análise de Retenção de Motoristas",
                style={
                    'textAlign': 'center',
                    'color': self.colors['text'],
                    'fontSize': '32px',
                    'fontWeight': '600',
                    'marginBottom': '40px'
                }
            ),
            self._create_metrics_section()
        ])
    
    def _create_metrics_section(self):
        """Cria a seção de métricas principais."""
        return html.Div([
            html.Div([
                self._create_metric_card(
                    "Taxa de Evasão",
                    f"{self.churn_metrics['churn_rate']:.1%}",
                    "vs. mês anterior",
                    "↑ 2.3%",
                    self.colors['danger']
                ),
                self._create_metric_card(
                    "Tempo Médio até Evasão",
                    f"{self.churn_metrics['avg_days_to_churn']:.0f} dias",
                    "Oportunidade de ação",
                    "30 dias críticos",
                    self.colors['warning']
                ),
                self._create_metric_card(
                    "Receita em Risco",
                    f"R$ {self.churn_metrics['revenue_at_risk']:,.0f}",
                    "Potencial de recuperação",
                    "↑ 35% com retenção",
                    self.colors['primary']
                )
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '40px'})
        ])
    
    def _create_trends_section(self):
        """Cria a seção de tendências temporais."""
        return html.Div([
            html.H2("Tendência e Impacto Financeiro", style=self.subtitle_style),
            dcc.Graph(
                id='temporal-trend',
                figure=self._plot_temporal_trend(),
                config={'displayModeBar': True, 'displaylogo': False}
            )
        ], style=self.card_style)
    
    def _create_segments_section(self):
        """Cria a seção de análise por segmentos."""
        return html.Div([
            html.H2("Perfil dos Motoristas por Segmento", style=self.subtitle_style),
            html.P(
                "Análise do comportamento e valor por segmento",
                style={'color': self.colors['text'], 'marginBottom': '20px'}
            ),
            dcc.Graph(
                id='segment-analysis',
                figure=self._plot_segment_analysis('kind'),
                config={'displayModeBar': True, 'displaylogo': False}
            )
        ], style=self.card_style)
    
    def _create_risk_section(self):
        """Cria a seção de indicadores de risco."""
        return html.Div([
            html.H2("Indicadores de Risco de Evasão", style=self.subtitle_style),
            html.P(
                "Padrões comportamentais que indicam propensão à evasão",
                style={'color': self.colors['text'], 'marginBottom': '20px'}
            ),
            dcc.Graph(
                id='behavioral-patterns',
                figure=self._plot_behavioral_patterns(),
                config={'displayModeBar': True, 'displaylogo': False}
            )
        ], style=self.card_style)
    
    def _create_strategic_recommendations_section(self):
        """Cria a seção de recomendações estratégicas."""
        return html.Div([
            html.H2("Recomendações Estratégicas", style=self.subtitle_style),
            html.P(
                "Ações prioritárias para redução da evasão",
                style={'color': self.colors['text'], 'marginBottom': '20px'}
            ),
            dcc.Graph(
                id='strategic-recommendations',
                figure=self._plot_strategic_recommendations(),
                config={'displayModeBar': True, 'displaylogo': False}
            )
        ], style=self.card_style)
    
    def _create_metric_card(self, title, value, subtitle, delta, color):
        """Cria um card de métrica com estilo padronizado."""
        return html.Div([
            html.H4(title, style={
                'fontFamily': self.plot_style['font_family'],
                'color': self.colors['text'],
                'fontSize': '14px',
                'fontWeight': '500',
                'marginBottom': '8px'
            }),
            html.H2(value, style={
                'fontFamily': self.plot_style['font_family'],
                'color': color,
                'fontSize': '32px',
                'fontWeight': '600',
                'marginBottom': '8px'
            }),
            html.Div([
                html.Span(subtitle, style={
                    'color': self.colors['text'],
                    'fontSize': '12px',
                    'marginRight': '8px'
                }),
                html.Span(delta, style={
                    'color': color,
                    'fontSize': '12px',
                    'fontWeight': '500'
                })
            ])
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'padding': '24px',
            'margin': '10px',
            'flex': '1',
            'textAlign': 'center',
            'transition': 'transform 0.2s ease',
            ':hover': {
                'transform': 'translateY(-2px)'
            }
        })
    
    def _setup_callbacks(self):
        """Configura os callbacks para interatividade."""
        @self.app.callback(
            Output('temporal-trend', 'figure'),
            Input('temporal-trend', 'id')
        )
        def update_temporal_trend(_):
            return self._plot_temporal_trend()
        
        @self.app.callback(
            Output('segment-analysis', 'figure'),
            Input('segment-selector', 'value')
        )
        def update_segment_analysis(segment):
            return self._plot_segment_analysis(segment or 'kind')
        
        @self.app.callback(
            Output('behavioral-patterns', 'figure'),
            Input('behavioral-patterns', 'id')
        )
        def update_behavioral_patterns(_):
            return self._plot_behavioral_patterns()
        
        if self.model_metrics:
            @self.app.callback(
                Output('model-performance', 'figure'),
                Input('model-performance', 'id')
            )
            def update_model_performance(_):
                return self._plot_model_performance()
    
    def _plot_temporal_trend(self):
        """Plota tendência temporal do churn com foco em impacto financeiro"""
        temporal_data = self.data.copy()
        date_column = 'transaction_date' if 'transaction_date' in temporal_data.columns else 'date'
        
        if date_column not in temporal_data.columns:
            return {}
        
        # Preparar dados
        temporal_data[date_column] = pd.to_datetime(temporal_data[date_column])
        
        # Calcular métricas mensais
        monthly_metrics = pd.DataFrame()
        
        # Agrupar por mês
        monthly_data = temporal_data.groupby(pd.Grouper(key=date_column, freq='M'))
        
        # Calcular motoristas ativos e receita por mês
        monthly_metrics['drivers'] = monthly_data['driver_id'].nunique()
        monthly_metrics['revenue'] = monthly_data['amount'].sum()
        
        # Calcular taxa de churn mês a mês
        months = sorted(temporal_data[date_column].dt.to_period('M').unique())
        
        for i in range(1, len(months)):
            current_month = months[i]
            prev_month = months[i-1]
            
            # Motoristas do mês anterior
            prev_drivers = set(temporal_data[
                temporal_data[date_column].dt.to_period('M') == prev_month
            ]['driver_id'])
            
            # Motoristas do mês atual
            current_drivers = set(temporal_data[
                temporal_data[date_column].dt.to_period('M') == current_month
            ]['driver_id'])
            
            # Motoristas que não continuaram
            churned = len(prev_drivers - current_drivers)
            
            # Taxa de churn
            month_date = current_month.to_timestamp()
            monthly_metrics.loc[month_date, 'churn_rate'] = (
                churned / len(prev_drivers) if prev_drivers else 0
            )
            
            # Calcular crescimento líquido
            monthly_metrics.loc[month_date, 'growth_rate'] = (
                (len(current_drivers) - len(prev_drivers)) / len(prev_drivers)
                if prev_drivers else 0
            )
        
        # Preencher primeiro mês com 0
        first_month = months[0].to_timestamp()
        monthly_metrics.loc[first_month, 'churn_rate'] = 0
        monthly_metrics.loc[first_month, 'growth_rate'] = 0
        
        # Calcular receita em risco
        monthly_metrics['revenue_at_risk'] = monthly_metrics['revenue'] * monthly_metrics['churn_rate']
        
        # Criar figura com subplots lado a lado
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Evolução da Taxa de Evasão',
                'Impacto na Receita e Volume de Motoristas'
            ),
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # 1. Taxa de Churn
        fig.add_trace(
            go.Bar(
                x=monthly_metrics.index,
                y=monthly_metrics['churn_rate'],
                name='Taxa de Evasão',
                marker_color=self.colors['danger'],
                opacity=0.7,
                width=20*24*60*60*1000,  # 20 dias em milissegundos
                hovertemplate='Período: %{x|%B %Y}<br>Taxa de Evasão: %{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Receita em Risco e Volume de Motoristas
        fig.add_trace(
            go.Bar(
                x=monthly_metrics.index,
                y=monthly_metrics['revenue_at_risk'],
                name='Receita em Risco',
                marker_color=self.colors['warning'],
                opacity=0.7,
                width=20*24*60*60*1000,
                hovertemplate='Período: %{x|%B %Y}<br>Receita em Risco: R$ %{y:,.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=monthly_metrics.index,
                y=monthly_metrics['drivers'],
                name='Total de Motoristas',
                marker_color=self.colors['primary'],
                opacity=0.7,
                width=20*24*60*60*1000,
                hovertemplate='Período: %{x|%B %Y}<br>Motoristas Ativos: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=2,
            secondary_y=True
        )
        
        # Atualizar layout
        fig.update_layout(
            height=500,
            template='plotly_white',
            font_family=self.plot_style['font_family'],
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            plot_bgcolor=self.plot_style['plot_bgcolor'],
            paper_bgcolor=self.plot_style['paper_bgcolor'],
            bargap=0.15,
            bargroupgap=0.1
        )
        
        # Atualizar eixos do primeiro gráfico
        fig.update_yaxes(
            title_text="Taxa de Evasão",
            tickformat='.0%',
            secondary_y=False,
            row=1, col=1,
            range=[0, max(monthly_metrics['churn_rate']) * 1.2],
            gridcolor='lightgray'
        )
        
        # Atualizar eixos do segundo gráfico
        fig.update_yaxes(
            title_text="Receita em Risco (R$)",
            tickformat=',.0f',
            secondary_y=False,
            row=1, col=2,
            gridcolor='lightgray'
        )
        fig.update_yaxes(
            title_text="Total de Motoristas",
            tickformat=',.0f',
            secondary_y=True,
            row=1, col=2,
            gridcolor='lightgray'
        )
        
        # Atualizar eixos X
        for i in [1]:
            fig.update_xaxes(
                title_text="Período",
                row=i, col=1,
                tickangle=45,
                dtick="M1",
                tickformat="%b/%Y",
                gridcolor='lightgray'
            )
            fig.update_xaxes(
                title_text="Período",
                row=i, col=2,
                tickangle=45,
                dtick="M1",
                tickformat="%b/%Y",
                gridcolor='lightgray'
            )
        
        return fig
    
    def _plot_segment_analysis(self, segment):
        """Plota análise por segmento"""
        data = self.data.copy()
        
        if segment not in data.columns:
            print(f"Aviso: Coluna {segment} não encontrada.")
            return {}
        
        # Preparar dados
        date_column = 'transaction_date' if 'transaction_date' in data.columns else 'date'
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Calcular métricas por segmento
        segment_metrics = data.groupby(segment).agg({
            'driver_id': 'nunique',
            'amount': ['sum', 'mean', 'count']
        })
        
        segment_metrics.columns = ['total_drivers', 'total_revenue', 'avg_amount', 'transaction_count']
        segment_metrics = segment_metrics.reset_index()
        
        # Calcular métricas adicionais
        segment_metrics['avg_transactions'] = segment_metrics['transaction_count'] / segment_metrics['total_drivers']
        segment_metrics['revenue_share'] = segment_metrics['total_revenue'] / segment_metrics['total_revenue'].sum()
        
        # Criar figura com subplots em matriz 2x2
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribuição de Motoristas',
                'Receita Total (R$)',
                'Ticket Médio (R$)',
                'Média de Transações'
            ),
            specs=[
                [{'type': 'domain'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.20,
            horizontal_spacing=0.15
        )

        # 1. Distribuição de Motoristas (Pizza)
        fig.add_trace(
            go.Pie(
                labels=segment_metrics[segment],
                values=segment_metrics['total_drivers'],
                hole=0.4,
                textinfo='percent+label',
                textposition='outside',
                texttemplate='%{label}<br>%{percent}',
                marker=dict(
                    colors=[self.colors['primary'], self.colors['success'], 
                           self.colors['warning'], self.colors['info']]
                ),
                hovertemplate=(
                    'Segmento: %{label}<br>' +
                    'Motoristas: %{value:,.0f}<br>' +
                    'Proporção: %{percent}<extra></extra>'
                ),
                # domain={'x': [0.05, 0.45], 'y': [0.52, 0.98]}
            ),
            row=1, col=1
        )

        # 2. Receita Total (Barras)
        fig.add_trace(
            go.Bar(
                x=segment_metrics[segment],
                y=segment_metrics['total_revenue'],
                marker_color=self.colors['primary'],
                text=segment_metrics['revenue_share'].apply(lambda x: f'{x:.1%}'),
                textposition='auto',
                hovertemplate=(
                    'Segmento: %{x}<br>' +
                    'Receita Total: R$ %{y:,.2f}<br>' +
                    'Participação: %{text}<extra></extra>'
                )
            ),
            row=1, col=2
        )

        # 3. Ticket Médio (Barras)
        fig.add_trace(
            go.Bar(
                x=segment_metrics[segment],
                y=segment_metrics['avg_amount'],
                marker_color=self.colors['success'],
                text=segment_metrics['avg_amount'].apply(lambda x: f'R$ {x:,.0f}'),
                textposition='auto',
                hovertemplate=(
                    'Segmento: %{x}<br>' +
                    'Ticket Médio: R$ %{y:,.2f}<extra></extra>'
                )
            ),
            row=2, col=1
        )

        # 4. Média de Transações (Barras)
        fig.add_trace(
            go.Bar(
                x=segment_metrics[segment],
                y=segment_metrics['avg_transactions'],
                marker_color=self.colors['warning'],
                text=segment_metrics['avg_transactions'].apply(lambda x: f'{x:.1f}'),
                textposition='auto',
                hovertemplate=(
                    'Segmento: %{x}<br>' +
                    'Média de Transações: %{y:.1f}<extra></extra>'
                )
            ),
            row=2, col=2
        )

        # Atualizar layout
        fig.update_layout(
            height=800,
            template='plotly_white',
            showlegend=False,
            bargap=0.2,
            title=dict(
                text="Análise de Segmentos",
                font=dict(size=24),
                x=0.5,
                xanchor='center'
            ),
        )
          

        # Atualizar eixos
        # for i in [1, 2]:  # Linhas
        #     for j in [1, 2]:  # Colunas
        #         if not (i == 1 and j == 1):  # Pular o gráfico de pizza
        #             fig.update_xaxes(
        #                 title_text="Segmento",
        #                 row=i, col=j,
        #                 tickangle=45,
        #                 gridcolor='lightgray',
        #                 showgrid=True
        #             )
        #             fig.update_yaxes(
        #                 gridcolor='lightgray',
        #                 showgrid=True,
        #                 row=i, col=j
        #             )

        # Atualizar títulos dos eixos Y
        fig.update_yaxes(title_text=" ", row=1, col=1)
        fig.update_yaxes(title_text="Receita Total (R$)", row=1, col=2)
        fig.update_yaxes(title_text="Ticket Médio (R$)", row=2, col=1)
        fig.update_yaxes(title_text="Média de Transações", row=2, col=2)


        # Atualizar eixos com títulos mais claros
        fig.update_xaxes(title_text=" ", row=1, col=1)
        fig.update_xaxes(title_text=" ", row=1, col=2)
        fig.update_xaxes(title_text=" ", row=2, col=1)
        fig.update_xaxes(title_text=" ", row=2, col=2)
         
        return fig
    
    def _plot_behavioral_patterns(self):
        """Plota indicadores de risco de evasão"""
        data = self.data.copy()
        
        # Calcular métricas por motorista
        driver_metrics = {}
        
        # 1. Valor médio das transações
        driver_metrics['avg_amount'] = data.groupby('driver_id')['amount'].mean()
        
        # 2. Frequência de transações
        date_column = 'transaction_date' if 'transaction_date' in data.columns else 'date'
        data[date_column] = pd.to_datetime(data[date_column])
        data['month'] = pd.to_datetime(data[date_column]).dt.to_period('M')
        transactions_per_month = data.groupby(['driver_id', 'month']).size().reset_index()
        driver_metrics['freq'] = transactions_per_month.groupby('driver_id')[0].mean()
        
        # 3. Variabilidade nos valores
        driver_metrics['amount_std'] = data.groupby('driver_id')['amount'].std().fillna(0)
        
        # 4. Dias desde última transação
        last_date = pd.to_datetime(data[date_column]).max()
        last_transaction = data.groupby('driver_id')[date_column].max()
        driver_metrics['days_since_last'] = (last_date - pd.to_datetime(last_transaction)).dt.days
        
        # Criar figura
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Ticket Médio por Motorista',
                'Frequência de Viagens por Mês',
                'Variação no Valor das Viagens',
                'Dias Sem Realizar Viagens'
            ),
            vertical_spacing=0.20,
            horizontal_spacing=0.15
        )
        
        # Configuração base para os histogramas
        hist_config = dict(
            nbinsx=20,
            opacity=0.8
        )
        
        # 1. Ticket Médio
        fig.add_trace(
            go.Histogram(
                x=driver_metrics['avg_amount'],
                name='Ticket Médio',
                marker=dict(
                    color=self.colors['primary'],
                    line=dict(color=self.colors['primary'], width=1)
                ),
                hovertemplate='Ticket Médio: R$ %{x:.2f}<br>Quantidade: %{y}<extra></extra>',
                **hist_config
            ),
            row=1, col=1
        )
        
        # 2. Frequência de Viagens
        fig.add_trace(
            go.Histogram(
                x=driver_metrics['freq'],
                name='Frequência',
                marker=dict(
                    color=self.colors['success'],
                    line=dict(color=self.colors['success'], width=1)
                ),
                hovertemplate='Viagens/Mês: %{x:.1f}<br>Quantidade: %{y}<extra></extra>',
                **hist_config
            ),
            row=1, col=2
        )
        
        # 3. Variação nos Valores
        fig.add_trace(
            go.Histogram(
                x=driver_metrics['amount_std'],
                name='Variação',
                marker=dict(
                    color=self.colors['warning'],
                    line=dict(color=self.colors['warning'], width=1)
                ),
                hovertemplate='Variação: R$ %{x:.2f}<br>Quantidade: %{y}<extra></extra>',
                **hist_config
            ),
            row=2, col=1
        )
        
        # 4. Dias Inativos
        fig.add_trace(
            go.Histogram(
                x=driver_metrics['days_since_last'],
                name='Dias Inativos',
                marker=dict(
                    color=self.colors['danger'],
                    line=dict(color=self.colors['danger'], width=1)
                ),
                hovertemplate='Dias Inativos: %{x:.0f}<br>Quantidade: %{y}<extra></extra>',
                **hist_config
            ),
            row=2, col=2
        )
        
        # Atualizar layout
        fig.update_layout(
            height=800,
            template='plotly_white',
            showlegend=False,
            bargap=0.2,  # Espaçamento entre barras no nível do layout
            title=dict(
                text="Indicadores de Risco de Evasão",
                font=dict(size=24),
                x=0.5,
                xanchor='center'
            )
        )
        
        # Atualizar eixos com títulos mais claros
        fig.update_xaxes(title_text="Valor Médio (R$)", row=1, col=1)
        fig.update_xaxes(title_text="Viagens por Mês", row=1, col=2)
        fig.update_xaxes(title_text="Variação no Valor (R$)", row=2, col=1)
        fig.update_xaxes(title_text="Dias Sem Viagens", row=2, col=2)
        
        fig.update_yaxes(title_text="Quantidade de Motoristas", row=1, col=1)
        fig.update_yaxes(title_text="Quantidade de Motoristas", row=1, col=2)
        fig.update_yaxes(title_text="Quantidade de Motoristas", row=2, col=1)
        fig.update_yaxes(title_text="Quantidade de Motoristas", row=2, col=2)
        
        return fig
    
    def _plot_model_performance(self):
        """Plota métricas de desempenho do modelo"""
        if not self.model_metrics:
            # Retornar figura vazia com mensagem
            fig = go.Figure()
            fig.add_annotation(
                text="Métricas do modelo não disponíveis",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    family=self.plot_style['font_family'],
                    size=16,
                    color=self.colors['text']
                )
            )
            fig.update_layout(
                height=400,
                template='plotly_white',
                paper_bgcolor=self.plot_style['paper_bgcolor']
            )
            return fig
        
        # Criar figura com subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Matriz de Confusão',
                'Curva ROC',
                'Importância das Features',
                'Distribuição de Probabilidades'
            ),
            vertical_spacing=0.22,
            horizontal_spacing=0.15
        )
        
        # 1. Matriz de Confusão
        if 'confusion_matrix' in self.model_metrics:
            cm = self.model_metrics['confusion_matrix']
            labels = ['Não Churned', 'Churned']
            
            heatmap = go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16, "family": self.plot_style['font_family']},
                colorscale=[[0, self.colors['gray']], [1, self.colors['primary']]],
                showscale=False,
                hoverongaps=False,
                hovertemplate=(
                    'Previsto: %{x}<br>' +
                    'Real: %{y}<br>' +
                    'Quantidade: %{text}<extra></extra>'
                )
            )
            fig.add_trace(heatmap, row=1, col=1)
        
        # 2. Curva ROC
        if all(key in self.model_metrics for key in ['fpr', 'tpr', 'roc_auc']):
            fig.add_trace(
                go.Scatter(
                    x=self.model_metrics['fpr'],
                    y=self.model_metrics['tpr'],
                    name=f'ROC (AUC = {self.model_metrics["roc_auc"]:.3f})',
                    line=dict(
                        color=self.colors['primary'],
                        width=2,
                        shape='spline'
                    ),
                    hovertemplate=(
                        'Taxa de Falsos Positivos: %{x:.3f}<br>' +
                        'Taxa de Verdadeiros Positivos: %{y:.3f}<extra></extra>'
                    )
                ),
                row=1, col=2
            )
            
            # Linha de referência
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    line=dict(
                        color=self.colors['gray'],
                        width=2,
                        dash='dash'
                    ),
                    showlegend=False,
                    hovertemplate='Linha de Base<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Importância das Features
        if all(key in self.model_metrics for key in ['feature_names', 'feature_importance']):
            feature_importance = pd.DataFrame({
                'feature': self.model_metrics['feature_names'],
                'importance': self.model_metrics['feature_importance']
            }).sort_values('importance', ascending=True).tail(10)
            
            fig.add_trace(
                go.Bar(
                    y=feature_importance['feature'],
                    x=feature_importance['importance'],
                    orientation='h',
                    marker=dict(
                        color=self.colors['success'],
                        line=dict(
                            color=self.colors['success'],
                            width=1
                        )
                    ),
                    hovertemplate=(
                        'Feature: %{y}<br>' +
                        'Importância: %{x:.3f}<extra></extra>'
                    )
                ),
                row=2, col=1
            )
        
        # 4. Distribuição de Probabilidades
        if 'y_prob' in self.model_metrics:
            fig.add_trace(
                go.Histogram(
                    x=self.model_metrics['y_prob'],
                    nbinsx=30,
                    name='Probabilidades',
                    marker=dict(
                        color=self.colors['warning'],
                        line=dict(
                            color=self.colors['warning'],
                            width=1
                        )
                    ),
                    hovertemplate=(
                        'Probabilidade: %{x:.2f}<br>' +
                        'Frequência: %{y}<extra></extra>'
                    )
                ),
                row=2, col=2
            )
        
        # Atualizar layout
        fig.update_layout(
            height=800,
            template='plotly_white',
            showlegend=True,
            font_family=self.plot_style['font_family'],
            title=dict(
                text="Análise de Desempenho do Modelo",
                font=dict(
                    size=self.plot_style['title_font_size'],
                    color=self.plot_style['title_font_color'],
                    family=self.plot_style['title_font_family']
                ),
                x=self.plot_style['title_x'],
                xanchor=self.plot_style['title_xanchor']
            ),
            legend=self.plot_style['legend'],
            plot_bgcolor=self.plot_style['plot_bgcolor'],
            paper_bgcolor=self.plot_style['paper_bgcolor']
        )
        
        # Atualizar eixos
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(
                    gridcolor=self.plot_style['xaxis']['gridcolor'],
                    linecolor=self.plot_style['xaxis']['linecolor'],
                    title_font=self.plot_style['xaxis']['title_font'],
                    tickfont=self.plot_style['xaxis']['tickfont'],
                    row=i, col=j
                )
                fig.update_yaxes(
                    gridcolor=self.plot_style['yaxis']['gridcolor'],
                    linecolor=self.plot_style['yaxis']['linecolor'],
                    title_font=self.plot_style['yaxis']['title_font'],
                    tickfont=self.plot_style['yaxis']['tickfont'],
                    row=i, col=j
                )
        
        # Atualizar títulos dos eixos
        fig.update_xaxes(title_text="Previsto", row=1, col=1)
        fig.update_yaxes(title_text="Real", row=1, col=1)
        
        fig.update_xaxes(title_text="Taxa de Falsos Positivos", row=1, col=2)
        fig.update_yaxes(title_text="Taxa de Verdadeiros Positivos", row=1, col=2)
        
        fig.update_xaxes(title_text="Importância", row=2, col=1)
        fig.update_yaxes(title_text="Feature", row=2, col=1)
        
        fig.update_xaxes(title_text="Probabilidade de Churn", row=2, col=2)
        fig.update_yaxes(title_text="Frequência", row=2, col=2)
        
        return fig
    
    def _plot_strategic_recommendations(self):
        """Plota recomendações estratégicas focadas em ação"""
        data = self.data.copy()
        date_column = 'transaction_date' if 'transaction_date' in data.columns else 'date'
        
        # Preparar dados por segmento
        segment_metrics = {}
        data[date_column] = pd.to_datetime(data[date_column])
        data['month'] = pd.to_datetime(data[date_column]).dt.to_period('M')
        
        segments = data['kind'].unique()
        for segment in segments:
            segment_data = data[data['kind'] == segment]
            
            # Métricas mensais
            monthly_value = segment_data.groupby('month')['amount'].sum().mean()
            n_drivers = segment_data['driver_id'].nunique()
            
            # Churn rate
            last_date = pd.to_datetime(data[date_column]).max()
            last_transactions = segment_data.groupby('driver_id')[date_column].max()
            churned = (last_date - pd.to_datetime(last_transactions)).dt.days > 30
            churn_rate = churned.mean()
            
            # Valor médio por motorista
            avg_value_per_driver = monthly_value / n_drivers
            
            segment_metrics[segment] = {
                'monthly_value': monthly_value,
                'n_drivers': n_drivers,
                'churn_rate': churn_rate,
                'avg_value_per_driver': avg_value_per_driver,
                'revenue_at_risk': monthly_value * churn_rate
            }
        
        # Criar DataFrame
        segment_df = pd.DataFrame.from_dict(segment_metrics, orient='index')
        
        # Criar figura com foco em ações estratégicas
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Priorização de Segmentos',
                'Potencial de Recuperação de Receita',
                'Valor por Motorista vs. Risco de Evasão',
                'Distribuição do Impacto Financeiro'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "domain"}]
            ],
            vertical_spacing=0.20,
            horizontal_spacing=0.15
        )
        
        # 1. Priorização de Segmentos
        priority_score = segment_df['revenue_at_risk'] * segment_df['churn_rate']
        priority_df = pd.DataFrame({
            'segment': priority_score.index,
            'score': priority_score.values
        }).sort_values('score', ascending=True)
        
        fig.add_trace(
            go.Bar(
                y=priority_df['segment'],
                x=priority_df['score'],
                orientation='h',
                marker_color=self.colors['primary'],
                hovertemplate=(
                    'Segmento: %{y}<br>' +
                    'Score de Prioridade: %{x:.2f}<br>' +
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )
        
        # 2. Potencial de Recuperação
        fig.add_trace(
            go.Bar(
                x=segment_df.index,
                y=segment_df['revenue_at_risk'],
                marker_color=self.colors['success'],
                hovertemplate=(
                    'Segmento: %{x}<br>' +
                    'Receita Recuperável: R$ %{y:,.2f}<br>' +
                    '<extra></extra>'
                )
            ),
            row=1, col=2
        )
        
        # 3. Matriz de Valor vs. Risco
        fig.add_trace(
            go.Scatter(
                x=segment_df['churn_rate'],
                y=segment_df['avg_value_per_driver'],
                mode='markers+text',
                marker=dict(
                    size=segment_df['n_drivers'] / segment_df['n_drivers'].max() * 50,
                    color=self.colors['warning'],
                    opacity=0.7
                ),
                text=segment_df.index,
                textposition="top center",
                hovertemplate=(
                    'Segmento: %{text}<br>' +
                    'Taxa de Evasão: %{x:.1%}<br>' +
                    'Valor por Motorista: R$ %{y:,.2f}<br>' +
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )
        
        # 4. Distribuição do Impacto
        fig.add_trace(
            go.Pie(
                labels=segment_df.index,
                values=segment_df['revenue_at_risk'],
                hole=0.4,
                marker=dict(colors=[
                    self.colors['primary'],
                    self.colors['success'],
                    self.colors['warning'],
                    self.colors['danger']
                ]),
                hovertemplate=(
                    'Segmento: %{label}<br>' +
                    'Impacto: R$ %{value:,.2f}<br>' +
                    'Proporção: %{percent}<br>' +
                    '<extra></extra>'
                )
            ),
            row=2, col=2
        )
        
        # Atualizar layout
        fig.update_layout(
            height=800,
            template='plotly_white',
            showlegend=False,
            title=dict(
                text="Recomendações Estratégicas por Segmento",
                font=dict(size=24),
                x=0.5,
                xanchor='center'
            )
        )
        
        # Atualizar eixos com títulos mais claros
        fig.update_xaxes(title_text="Score de Prioridade", row=1, col=1)
        fig.update_xaxes(title_text="Segmento", row=1, col=2)
        fig.update_xaxes(title_text="Taxa de Evasão", tickformat='.0%', row=2, col=1)
        
        fig.update_yaxes(title_text="Segmento", row=1, col=1)
        fig.update_yaxes(title_text="Receita Recuperável (R$)", row=1, col=2)
        fig.update_yaxes(title_text="Valor por Motorista (R$)", row=2, col=1)
        
        return fig
    
    def run_server(self, debug=True, port=8050):
        """Inicia o servidor do dashboard."""
        self.app.run_server(debug=debug, port=port)

if __name__ == '__main__':
    # Exemplo de uso
    import numpy as np
    
    # Dados de exemplo
    np.random.seed(42)
    n_drivers = 1000
    n_days = 90
    
    dates = pd.date_range(end=datetime.now(), periods=n_days)
    drivers = range(1, n_drivers + 1)
    
    data = {
        'date': np.repeat(dates, n_drivers),
        'driver_id': np.tile(drivers, n_days),
        'amount': np.random.normal(700, 100, n_drivers * n_days),
        'kind': np.random.choice(
            ['FIRST_PAYMENT', 'RECURRENCY', 'FIRST_PAYMENT_EXCHANGE'],
            n_drivers * n_days,
            p=[0.3, 0.5, 0.2]
        )
    }
    
    df = pd.DataFrame(data)
    
    # Métricas de exemplo
    churn_metrics = {
        'churn_rate': 0.15,
        'avg_days_to_churn': 45,
        'revenue_at_risk': 1500000
    }
    
    # Iniciar dashboard
    dashboard = ChurnDashboard(df, churn_metrics)
    dashboard.run_server(debug=True) 