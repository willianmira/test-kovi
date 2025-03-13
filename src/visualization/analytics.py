import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import os

class ChurnAnalytics:
    def __init__(self, data, model=None, y_true=None, y_pred=None, y_proba=None):
        self.data = data
        self.model = model
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        
    def generate_eda_plots(self, output_dir='reports/figures'):
        """Gera visualizações exploratórias avançadas dos dados"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Análise Temporal
        self._plot_temporal_patterns(output_dir)
        
        # 2. Análise de Segmentação
        self._plot_segment_analysis(output_dir)
        
        # 3. Análise de Comportamento
        self._plot_behavioral_patterns(output_dir)
        
        # 4. Correlações e Padrões
        self._plot_correlation_patterns(output_dir)
    
    def _plot_temporal_patterns(self, output_dir):
        """Análise de padrões temporais"""
        # Tendência de churn ao longo do tempo
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Tendência de Churn', 'Sazonalidade'))
        
        # Preparar dados temporais
        temporal_data = self.data.copy()
        
        # Verificar qual coluna de data está disponível
        date_column = 'transaction_date' if 'transaction_date' in temporal_data.columns else 'date'
        if date_column not in temporal_data.columns:
            print("Aviso: Nenhuma coluna de data encontrada. Pulando análise temporal.")
            return
            
        # Garantir que a data está no formato correto
        if not pd.api.types.is_datetime64_any_dtype(temporal_data[date_column]):
            temporal_data[date_column] = pd.to_datetime(temporal_data[date_column])
            
        # Calcular churn por mês
        temporal_data = temporal_data.set_index(date_column)
        
        # Calcular número de motoristas únicos por mês
        monthly_drivers = temporal_data.resample('M')['driver_id'].nunique()
        
        # Calcular taxa de retenção (inverso do churn)
        retention_rate = monthly_drivers / monthly_drivers.shift(1)
        churn_rate = 1 - retention_rate.fillna(0)
        
        # Plotar tendência de churn
        fig.add_trace(
            go.Scatter(x=churn_rate.index, y=churn_rate.values,
                      name='Taxa de Churn Mensal'),
            row=1, col=1
        )
        
        # Análise de sazonalidade (usando número de transações)
        seasonal = temporal_data.groupby(temporal_data.index.month).size()
        seasonal = seasonal / seasonal.max()  # Normalizar
        
        fig.add_trace(
            go.Bar(x=seasonal.index, y=seasonal.values,
                  name='Sazonalidade'),
            row=2, col=1
        )
        
        # Atualizar layout
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Análise Temporal de Churn"
        )
        
        fig.update_xaxes(title_text="Data", row=1, col=1)
        fig.update_xaxes(title_text="Mês", row=2, col=1)
        fig.update_yaxes(title_text="Taxa de Churn", row=1, col=1)
        fig.update_yaxes(title_text="Transações Normalizadas", row=2, col=1)
        
        # Salvar figura
        fig.write_html(f'{output_dir}/temporal_analysis.html')
    
    def _plot_segment_analysis(self, output_dir):
        """Análise de segmentação"""
        # Preparar dados
        data = self.data.copy()
        
        # Verificar se temos a coluna 'kind' para segmentação
        if 'kind' not in data.columns:
            print("Aviso: Coluna 'kind' não encontrada. Pulando análise de segmentação.")
            return
        
        # Calcular churn por tipo de pagamento
        # Primeiro, identificar últimas transações por motorista
        last_transactions = data.groupby('driver_id').agg({
            'transaction_date': 'max',
            'kind': 'last'  # Pegar o último tipo de pagamento
        })
        
        # Calcular churn (30 dias sem transação)
        reference_date = data['transaction_date'].max()
        last_transactions['churn'] = (
            reference_date - last_transactions['transaction_date']
        ).dt.days > 30
        
        # Calcular métricas por tipo de pagamento
        segment_data = last_transactions.groupby('kind').agg({
            'churn': ['mean', 'count']
        })
        
        segment_data.columns = ['churn_rate', 'total_drivers']
        segment_data = segment_data.reset_index()
        
        # Criar gráfico de barras para taxa de churn
        fig_churn = go.Figure()
        fig_churn.add_trace(
            go.Bar(x=segment_data['kind'],
                   y=segment_data['churn_rate'],
                   name='Taxa de Churn')
        )
        
        fig_churn.update_layout(
            title='Taxa de Churn por Tipo de Pagamento',
            xaxis_title='Tipo de Pagamento',
            yaxis_title='Taxa de Churn',
            height=400,
            showlegend=True
        )
        
        # Criar gráfico de pizza para distribuição
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Pie(labels=segment_data['kind'],
                   values=segment_data['total_drivers'],
                   name='Distribuição')
        )
        
        fig_dist.update_layout(
            title='Distribuição de Motoristas por Tipo de Pagamento',
            height=400,
            showlegend=True
        )
        
        # Salvar figuras
        fig_churn.write_html(f"{output_dir}/segment_churn.html")
        fig_dist.write_html(f"{output_dir}/segment_distribution.html")
    
    def _plot_behavioral_patterns(self, output_dir):
        """Análise de padrões comportamentais"""
        # Criar subplots para diferentes aspectos comportamentais
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Padrão de Uso',
                                         'Interações',
                                         'Perfil de Pagamento',
                                         'Indicadores de Risco'))
        
        # Adicionar visualizações comportamentais
        behavioral_metrics = self._calculate_behavioral_metrics()
        
        for i, (metric, values) in enumerate(behavioral_metrics.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Box(y=values, name=metric),
                row=row, col=col
            )
        
        fig.update_layout(height=800,
                         title_text="Análise de Padrões Comportamentais")
        fig.write_html(f'{output_dir}/behavioral_analysis.html')
    
    def _plot_correlation_patterns(self, output_dir):
        """Análise de correlações e padrões"""
        # Matriz de correlação interativa
        corr_matrix = self.data.select_dtypes(include=['float64', 'int64']).corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title='Matriz de Correlação Interativa',
            height=800,
            width=800
        )
        
        fig.write_html(f'{output_dir}/correlation_analysis.html')
    
    def generate_model_analysis(self, output_dir='reports/model_analysis'):
        """Gera visualizações avançadas da performance do modelo"""
        if not all([self.y_true is not None, self.y_pred is not None,
                   self.y_proba is not None]):
            raise ValueError("Dados de predição não fornecidos")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Curvas de Performance
        self._plot_model_curves(output_dir)
        
        # 2. Análise de Erros
        self._plot_error_analysis(output_dir)
        
        # 3. Calibração do Modelo
        self._plot_model_calibration(output_dir)
    
    def _plot_model_curves(self, output_dir):
        """Plota curvas de performance do modelo"""
        # Criar subplots para ROC e Precision-Recall
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Curva ROC',
                                         'Curva Precision-Recall'))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name='ROC',
                      fill='tozeroy'),
            row=1, col=1
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_true,
                                                    self.y_proba)
        fig.add_trace(
            go.Scatter(x=recall, y=precision,
                      name='Precision-Recall',
                      fill='tozeroy'),
            row=1, col=2
        )
        
        fig.update_layout(height=500,
                         title_text="Curvas de Performance do Modelo")
        fig.write_html(f'{output_dir}/model_curves.html')
    
    def _plot_error_analysis(self, output_dir):
        """Análise detalhada dos erros do modelo"""
        # Matriz de confusão interativa
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Previsto Negativo', 'Previsto Positivo'],
            y=['Real Negativo', 'Real Positivo'],
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Matriz de Confusão Interativa',
            height=600,
            width=600
        )
        
        fig.write_html(f'{output_dir}/confusion_matrix.html')
    
    def _plot_model_calibration(self, output_dir):
        """Plota a calibração do modelo"""
        # Criar bins de probabilidade prevista
        bins = pd.qcut(self.y_proba, q=10, duplicates='drop')
        
        # Calcular taxa real de churn por bin
        calibration_data = pd.DataFrame({
            'prob_pred': self.y_proba,
            'actual': self.y_true,
            'bin': bins
        })
        
        bin_stats = calibration_data.groupby('bin').agg({
            'prob_pred': 'mean',
            'actual': 'mean'
        }).reset_index()
        
        # Criar gráfico de calibração
        fig = go.Figure()
        
        # Linha de referência (calibração perfeita)
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1],
                      line=dict(dash='dash', color='gray'),
                      name='Calibração Perfeita')
        )
        
        # Pontos de calibração real
        fig.add_trace(
            go.Scatter(x=bin_stats['prob_pred'],
                      y=bin_stats['actual'],
                      mode='markers+lines',
                      name='Calibração Real')
        )
        
        fig.update_layout(
            title='Calibração do Modelo',
            xaxis_title='Probabilidade Prevista',
            yaxis_title='Taxa Real de Churn',
            height=500,
            showlegend=True
        )
        
        fig.write_html(f'{output_dir}/model_calibration.html')
    
    def generate_feature_importance(self, output_dir='reports/feature_importance'):
        """Gera visualização avançada da importância das features"""
        if self.model is None:
            raise ValueError("Modelo não fornecido")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Tentar encontrar o modelo no pipeline
        model = None
        feature_names = None
        
        # Procurar por modelos conhecidos no pipeline
        for step_name, step in self.model.named_steps.items():
            if hasattr(step, 'feature_importances_'):
                model = step
                break
        
        if model is None:
            raise ValueError("Não foi possível encontrar um modelo com feature importances no pipeline")
            
        # Tentar obter os nomes das features
        try:
            # Tentar pegar do embedding
            if 'embedding' in self.model.named_steps:
                feature_names = self.model.named_steps['embedding'].get_feature_names()
            else:
                # Criar nomes genéricos
                n_features = len(model.feature_importances_)
                feature_names = [f'Feature {i+1}' for i in range(n_features)]
        except:
            # Se falhar, usar nomes genéricos
            n_features = len(model.feature_importances_)
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        # Criar DataFrame com importância das features
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Criar visualização interativa
        fig = go.Figure()
        
        # Adicionar barras horizontais
        fig.add_trace(
            go.Bar(
                x=feature_importance['importance'].tail(20),
                y=feature_importance['feature'].tail(20),
                orientation='h',
                marker=dict(
                    color=feature_importance['importance'].tail(20),
                    colorscale='Viridis'
                )
            )
        )
        
        fig.update_layout(
            title='Top 20 Features Mais Importantes',
            xaxis_title='Importância Relativa',
            yaxis_title='Feature',
            height=800,
            showlegend=False
        )
        
        fig.write_html(f'{output_dir}/feature_importance.html')
        
        return feature_importance
    
    def _calculate_rfm_metrics(self):
        """Calcula métricas RFM (Recência, Frequência, Monetário) para segmentação"""
        try:
            # Preparar dados
            data = self.data.copy()
            
            # Verificar qual coluna de data está disponível
            date_column = 'transaction_date' if 'transaction_date' in data.columns else 'date'
            if date_column not in data.columns:
                print("Aviso: Nenhuma coluna de data encontrada. Pulando análise RFM.")
                return None
                
            # Garantir que a data está no formato correto
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                data[date_column] = pd.to_datetime(data[date_column])
            
            # Data de referência (última data no dataset + 1 dia)
            reference_date = data[date_column].max() + pd.Timedelta(days=1)
            
            # Calcular métricas por motorista
            rfm = data.groupby('driver_id').agg({
                date_column: lambda x: (reference_date - x.max()).days,  # Recência
                'amount': ['count', 'sum']  # Frequência e Monetário
            }).reset_index()
            
            # Renomear colunas
            rfm.columns = ['driver_id', 'recency', 'frequency', 'monetary']
            
            # Calcular scores (1-5) para cada métrica
            for metric in ['recency', 'frequency', 'monetary']:
                if metric == 'recency':
                    # Para recência, menor é melhor
                    rfm[f'{metric}_score'] = pd.qcut(rfm[metric], q=5, labels=False) + 1
                    rfm[f'{metric}_score'] = 6 - rfm[f'{metric}_score']  # Inverter scores
                else:
                    # Para frequência e monetário, maior é melhor
                    rfm[f'{metric}_score'] = pd.qcut(rfm[metric], q=5, labels=False) + 1
            
            # Calcular score RFM combinado (já são inteiros agora)
            rfm['rfm_score'] = (
                rfm['recency_score'] * 100 +
                rfm['frequency_score'] * 10 +
                rfm['monetary_score']
            )
            
            # Calcular risco de churn (baseado principalmente na recência)
            rfm['churn_risk'] = (6 - rfm['recency_score']) / 5  # Normalizado entre 0 e 1
            
            return rfm
            
        except Exception as e:
            print(f"Erro ao calcular métricas RFM: {str(e)}")
            return None
    
    def _calculate_behavioral_metrics(self):
        """Calcula métricas comportamentais dos motoristas"""
        try:
            # Preparar dados
            data = self.data.copy()
            
            # Verificar qual coluna de data está disponível
            date_column = 'transaction_date' if 'transaction_date' in data.columns else 'date'
            if date_column not in data.columns:
                print("Aviso: Nenhuma coluna de data encontrada. Pulando análise comportamental.")
                return None
                
            # Garantir que a data está no formato correto
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                data[date_column] = pd.to_datetime(data[date_column])
            
            # Calcular métricas por motorista
            metrics = {}
            
            # 1. Padrão de Uso: Valor médio das transações
            metrics['Valor Médio por Transação'] = data.groupby('driver_id')['amount'].mean()
            
            # 2. Interações: Número de transações por mês
            data['month'] = data[date_column].dt.to_period('M')
            transactions_per_month = data.groupby(['driver_id', 'month']).size().reset_index()
            metrics['Transações por Mês'] = transactions_per_month.groupby('driver_id')[0].mean()
            
            # 3. Perfil de Pagamento: Variação nos valores das transações
            metrics['Variação nos Valores'] = data.groupby('driver_id')['amount'].std().fillna(0)
            
            # 4. Indicadores de Risco: Dias desde a última transação
            last_date = data[date_column].max()
            last_transaction = data.groupby('driver_id')[date_column].max()
            metrics['Dias Desde Última Transação'] = (last_date - last_transaction).dt.days
            
            return metrics
            
        except Exception as e:
            print(f"Erro ao calcular métricas comportamentais: {str(e)}")
            return None 