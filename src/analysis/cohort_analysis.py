import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CohortAnalysis:
    def __init__(self, data):
        self.data = data
        self.cohort_data = None
        self.retention_matrix = None
        self.ltv_matrix = None
        
    def prepare_cohort_data(self):
        """Prepara dados para análise de cohort"""
        # Converter datas e renomear coluna se necessário
        if 'date' in self.data.columns:
            self.data = self.data.rename(columns={'date': 'transaction_date'})
        self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'])
        
        # Identificar primeira transação de cada motorista
        first_transactions = self.data.groupby('driver_id')['transaction_date'].min().reset_index()
        first_transactions['cohort_date'] = first_transactions['transaction_date'].dt.to_period('M')
        
        # Adicionar informação de cohort ao dataset principal
        self.data = self.data.merge(
            first_transactions[['driver_id', 'cohort_date']],
            on='driver_id',
            how='left'
        )
        
        # Calcular período relativo para cada transação
        self.data['period_number'] = (
            self.data['transaction_date'].dt.to_period('M') -
            self.data['cohort_date']
        ).apply(lambda x: x.n)
        
        return self.data
    
    def calculate_retention_matrix(self):
        """Calcula matriz de retenção por cohort"""
        # Agrupar por cohort e período
        cohort_data = self.data.groupby(['cohort_date', 'period_number'])['driver_id'].nunique().reset_index()
        
        # Criar matriz de retenção
        cohort_pivot = cohort_data.pivot(
            index='cohort_date',
            columns='period_number',
            values='driver_id'
        )
        
        # Calcular taxas de retenção
        self.retention_matrix = cohort_pivot.div(cohort_pivot[0], axis=0)
        
        return self.retention_matrix
    
    def calculate_ltv_matrix(self):
        """Calcula matriz de valor do cliente por cohort"""
        # Agrupar valor total por cohort e período
        ltv_data = self.data.groupby(['cohort_date', 'period_number'])['amount'].sum().reset_index()
        
        # Criar matriz de LTV
        self.ltv_matrix = ltv_data.pivot(
            index='cohort_date',
            columns='period_number',
            values='amount'
        )
        
        return self.ltv_matrix
    
    def plot_cohort_analysis(self, output_dir='reports/cohort_analysis'):
        """Gera visualizações da análise de cohort"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Mapa de Calor de Retenção
        self._plot_retention_heatmap(output_dir)
        
        # 2. Curvas de Retenção
        self._plot_retention_curves(output_dir)
        
        # 3. Análise de LTV
        self._plot_ltv_analysis(output_dir)
        
        # 4. Análise Combinada
        self._plot_combined_analysis(output_dir)
    
    def _plot_retention_heatmap(self, output_dir):
        """Plota mapa de calor de retenção"""
        if self.retention_matrix is None:
            self.calculate_retention_matrix()
        
        fig = go.Figure(data=go.Heatmap(
            z=self.retention_matrix.values,
            x=self.retention_matrix.columns,
            y=self.retention_matrix.index.astype(str),
            colorscale='RdYlBu_r',
            zmin=0,
            zmax=1
        ))
        
        fig.update_layout(
            title='Mapa de Calor de Retenção por Cohort',
            xaxis_title='Período',
            yaxis_title='Cohort',
            height=600
        )
        
        fig.write_html(f'{output_dir}/retention_heatmap.html')
    
    def _plot_retention_curves(self, output_dir):
        """Plota curvas de retenção"""
        if self.retention_matrix is None:
            self.calculate_retention_matrix()
        
        fig = go.Figure()
        
        for cohort in self.retention_matrix.index:
            fig.add_trace(
                go.Scatter(
                    x=self.retention_matrix.columns,
                    y=self.retention_matrix.loc[cohort],
                    name=str(cohort),
                    mode='lines+markers'
                )
            )
        
        fig.update_layout(
            title='Curvas de Retenção por Cohort',
            xaxis_title='Período',
            yaxis_title='Taxa de Retenção',
            height=500
        )
        
        fig.write_html(f'{output_dir}/retention_curves.html')
    
    def _plot_ltv_analysis(self, output_dir):
        """Plota análise de LTV"""
        if self.ltv_matrix is None:
            self.calculate_ltv_matrix()
        
        # Criar subplots
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('LTV Acumulado por Cohort',
                                         'LTV Médio por Período'))
        
        # LTV Acumulado
        cumulative_ltv = self.ltv_matrix.cumsum(axis=1)
        for cohort in cumulative_ltv.index:
            fig.add_trace(
                go.Scatter(
                    x=cumulative_ltv.columns,
                    y=cumulative_ltv.loc[cohort],
                    name=str(cohort),
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # LTV Médio por Período
        avg_ltv = self.ltv_matrix.mean()
        fig.add_trace(
            go.Bar(
                x=avg_ltv.index,
                y=avg_ltv.values,
                name='LTV Médio'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=800)
        fig.write_html(f'{output_dir}/ltv_analysis.html')
    
    def _plot_combined_analysis(self, output_dir):
        """Plota análise combinada de retenção e LTV"""
        # Criar figura com subplots
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Retenção vs LTV',
                                         'Valor por Cliente Retido'))
        
        # Retenção vs LTV
        retention_avg = self.retention_matrix.mean()
        ltv_avg = self.ltv_matrix.mean()
        
        fig.add_trace(
            go.Scatter(
                x=retention_avg.index,
                y=retention_avg.values,
                name='Retenção Média',
                yaxis='y1'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=ltv_avg.index,
                y=ltv_avg.values,
                name='LTV Médio',
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Valor por Cliente Retido
        value_per_retained = (
            self.ltv_matrix / 
            (self.retention_matrix * self.ltv_matrix.iloc[:, 0].values.reshape(-1, 1))
        ).mean()
        
        fig.add_trace(
            go.Scatter(
                x=value_per_retained.index,
                y=value_per_retained.values,
                name='Valor por Cliente Retido',
                mode='lines+markers'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=True
        )
        
        fig.write_html(f'{output_dir}/combined_analysis.html') 