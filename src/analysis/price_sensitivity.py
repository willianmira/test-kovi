import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class PriceSensitivityAnalysis:
    def __init__(self, data):
        self.data = data.copy()
        
        # Criar coluna de motoristas ativos
        self._prepare_data()
        
        self.elasticity = None
        self.optimal_prices = None
        self.price_segments = None
        
    def _prepare_data(self):
        """Prepara dados para análise de sensibilidade de preços"""
        # Agrupar por motorista e data para contar transações
        daily_activity = self.data.groupby(['driver_id', pd.Grouper(key='transaction_date', freq='D')]).size().reset_index()
        daily_activity.columns = ['driver_id', 'transaction_date', 'transactions']
        
        # Considerar motorista ativo se fez pelo menos uma transação
        daily_activity['is_active'] = (daily_activity['transactions'] > 0).astype(int)
        
        # Juntar com dados originais
        self.data = self.data.merge(
            daily_activity[['driver_id', 'transaction_date', 'is_active']],
            on=['driver_id', 'transaction_date'],
            how='left'
        )
        
    def calculate_price_elasticity(self, price_col='amount', demand_col='active_drivers'):
        """Calcula elasticidade-preço da demanda"""
        # Agrupar dados por preço
        price_demand = self.data.groupby(price_col)[demand_col].mean().reset_index()
        
        # Calcular variações percentuais
        price_pct_change = price_demand[price_col].pct_change()
        demand_pct_change = price_demand[demand_col].pct_change()
        
        # Calcular elasticidade
        self.elasticity = demand_pct_change / price_pct_change
        
        return self.elasticity.mean()
    
    def find_optimal_prices(self, price_col='amount', revenue_col='revenue'):
        """Encontra preços ótimos por segmento"""
        # Calcular receita por preço
        price_revenue = self.data.groupby(price_col)[revenue_col].sum().reset_index()
        
        # Ajustar curva de receita
        coeffs = np.polyfit(price_revenue[price_col],
                          price_revenue[revenue_col], 2)
        
        # Encontrar ponto de máximo
        a, b, c = coeffs
        optimal_price = -b / (2 * a)
        
        self.optimal_prices = {
            'price': optimal_price,
            'estimated_revenue': a * optimal_price**2 + b * optimal_price + c
        }
        
        return self.optimal_prices
    
    def segment_price_sensitivity(self, segment_col, price_col='amount'):
        """Analisa sensibilidade de preço por segmento"""
        segments = self.data[segment_col].unique()
        self.price_segments = {}
        
        for segment in segments:
            segment_data = self.data[self.data[segment_col] == segment]
            
            # Calcular elasticidade do segmento
            elasticity = self.calculate_price_elasticity(
                price_col=price_col,
                demand_col='active_drivers'
            )
            
            # Encontrar preço ótimo para o segmento
            optimal_price = self.find_optimal_prices(
                price_col=price_col,
                revenue_col='revenue'
            )
            
            self.price_segments[segment] = {
                'elasticity': elasticity,
                'optimal_price': optimal_price
            }
        
        return self.price_segments
    
    def plot_price_analysis(self, output_dir='reports/price_analysis'):
        """Gera visualizações da análise de sensibilidade de preços"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Curva de Demanda
            self._plot_demand_curve(output_dir)
            
            # 2. Distribuição de Preços
            self._plot_price_distribution(output_dir)
            
            # 3. Análise Temporal de Preços
            self._plot_price_trends(output_dir)
            
        except Exception as e:
            print(f"Erro ao gerar visualizações de preços: {str(e)}")
    
    def _plot_demand_curve(self, output_dir):
        """Plota curva de demanda"""
        # Agrupar por valor e calcular média de motoristas ativos
        price_demand = self.data.groupby('amount')['is_active'].mean().reset_index()
        price_demand = price_demand.sort_values('amount')
        
        # Suavizar a curva usando média móvel
        window = min(5, len(price_demand))
        price_demand['smoothed_demand'] = price_demand['is_active'].rolling(window=window, center=True).mean()
        
        fig = go.Figure()
        
        # Adicionar pontos originais
        fig.add_trace(
            go.Scatter(
                x=price_demand['amount'],
                y=price_demand['is_active'],
                mode='markers',
                name='Dados Originais',
                marker=dict(size=5)
            )
        )
        
        # Adicionar linha suavizada
        fig.add_trace(
            go.Scatter(
                x=price_demand['amount'],
                y=price_demand['smoothed_demand'],
                mode='lines',
                name='Tendência',
                line=dict(width=2)
            )
        )
        
        fig.update_layout(
            title='Curva de Demanda por Preço',
            xaxis_title='Valor da Transação',
            yaxis_title='Taxa de Atividade dos Motoristas',
            height=500
        )
        
        fig.write_html(f'{output_dir}/demand_curve.html')
    
    def _plot_price_distribution(self, output_dir):
        """Plota distribuição dos preços"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=self.data['amount'],
                nbinsx=50,
                name='Distribuição de Preços'
            )
        )
        
        fig.update_layout(
            title='Distribuição dos Valores de Transação',
            xaxis_title='Valor da Transação',
            yaxis_title='Frequência',
            height=500
        )
        
        fig.write_html(f'{output_dir}/price_distribution.html')
    
    def _plot_price_trends(self, output_dir):
        """Plota tendências de preços ao longo do tempo"""
        # Calcular médias diárias
        daily_prices = self.data.groupby('transaction_date').agg({
            'amount': ['mean', 'std', 'count']
        }).reset_index()
        
        daily_prices.columns = ['date', 'mean_price', 'std_price', 'transactions']
        
        # Criar subplots
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Preço Médio Diário', 'Volume de Transações'))
        
        # Adicionar preço médio
        fig.add_trace(
            go.Scatter(
                x=daily_prices['date'],
                y=daily_prices['mean_price'],
                mode='lines',
                name='Preço Médio'
            ),
            row=1, col=1
        )
        
        # Adicionar volume de transações
        fig.add_trace(
            go.Bar(
                x=daily_prices['date'],
                y=daily_prices['transactions'],
                name='Volume'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text='Tendências de Preços ao Longo do Tempo',
            showlegend=True
        )
        
        fig.write_html(f'{output_dir}/price_trends.html')
    
    def _plot_elasticity_analysis(self, output_dir):
        """Plota análise de elasticidade"""
        if self.elasticity is None:
            self.calculate_price_elasticity()
        
        # Criar figura com subplots
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Distribuição da Elasticidade',
                                         'Elasticidade vs Preço'))
        
        # Distribuição da elasticidade
        fig.add_trace(
            go.Histogram(
                x=self.elasticity,
                name='Distribuição'
            ),
            row=1, col=1
        )
        
        # Elasticidade vs Preço
        fig.add_trace(
            go.Scatter(
                x=self.data['amount'],
                y=self.elasticity,
                mode='markers',
                name='Elasticidade'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500)
        fig.write_html(f'{output_dir}/elasticity_analysis.html')
    
    def _plot_price_optimization(self, output_dir):
        """Plota análise de otimização de preços"""
        if self.optimal_prices is None:
            self.find_optimal_prices()
        
        # Criar dados para curva de receita
        prices = np.linspace(
            self.data['amount'].min(),
            self.data['amount'].max(),
            100
        )
        
        # Calcular receita estimada
        a, b, c = np.polyfit(self.data['amount'],
                            self.data['revenue'], 2)
        revenue = a * prices**2 + b * prices + c
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar curva de receita
        fig.add_trace(
            go.Scatter(
                x=prices,
                y=revenue,
                mode='lines',
                name='Receita Estimada'
            )
        )
        
        # Adicionar ponto ótimo
        fig.add_trace(
            go.Scatter(
                x=[self.optimal_prices['price']],
                y=[self.optimal_prices['estimated_revenue']],
                mode='markers',
                name='Preço Ótimo',
                marker=dict(size=10, color='red')
            )
        )
        
        fig.update_layout(
            title='Otimização de Preços',
            xaxis_title='Preço',
            yaxis_title='Receita',
            height=500
        )
        
        fig.write_html(f'{output_dir}/price_optimization.html')
    
    def _plot_segment_analysis(self, output_dir):
        """Plota análise por segmento"""
        if self.price_segments is None:
            return
        
        # Criar figura com subplots
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Elasticidade por Segmento',
                                         'Preço Ótimo por Segmento'))
        
        # Dados para plot
        segments = list(self.price_segments.keys())
        elasticities = [data['elasticity'] for data in self.price_segments.values()]
        optimal_prices = [data['optimal_price']['price']
                        for data in self.price_segments.values()]
        
        # Elasticidade por segmento
        fig.add_trace(
            go.Bar(
                x=segments,
                y=elasticities,
                name='Elasticidade'
            ),
            row=1, col=1
        )
        
        # Preço ótimo por segmento
        fig.add_trace(
            go.Bar(
                x=segments,
                y=optimal_prices,
                name='Preço Ótimo'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=800)
        fig.write_html(f'{output_dir}/segment_analysis.html') 