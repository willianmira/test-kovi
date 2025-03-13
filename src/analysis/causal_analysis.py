import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from datetime import timedelta

class CausalAnalysis:
    def __init__(self, data, target='churn', window_days=30, fast_mode=True):
        self.data = data.copy()
        self.target = target
        self.window_days = window_days
        self.model = None
        self.shap_values = None
        self.feature_importance = None
        self.fast_mode = fast_mode
        
        # Criar coluna de churn se não existir
        if self.target not in self.data.columns:
            self._create_churn_column()
        
    def _create_churn_column(self):
        """Cria a coluna de churn baseada na última transação"""
        # Agrupar por motorista e calcular última transação
        last_transactions = self.data.groupby('driver_id')['transaction_date'].max()
        
        # Data de referência (última data no dataset)
        reference_date = self.data['transaction_date'].max()
        
        # Identificar churned drivers (sem transações nos últimos X dias)
        churned_drivers = last_transactions[
            last_transactions < (reference_date - timedelta(days=self.window_days))
        ].index
        
        # Criar coluna de churn
        self.data[self.target] = self.data['driver_id'].isin(churned_drivers).astype(int)
        
    def prepare_data(self):
        """Prepara dados para análise causal"""
        try:
            # Verificar se temos as colunas necessárias
            numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
            numeric_features = numeric_features.drop(self.target) if self.target in numeric_features else numeric_features
            
            if len(numeric_features) == 0:
                raise ValueError("Não há features numéricas disponíveis para análise")
            
            # Separar features e target
            X = self.data[numeric_features].copy()
            y = self.data[self.target]
            
            # Escalonar features numéricas
            scaler = StandardScaler()
            X = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            return X, y
            
        except Exception as e:
            print(f"Erro ao preparar dados: {str(e)}")
            return None, None
        
    def fit_model(self):
        """Treina modelo para análise causal"""
        try:
            X, y = self.prepare_data()
            if X is None or y is None:
                return None
            
            # Treinar Random Forest com configurações otimizadas
            self.model = RandomForestClassifier(
                n_estimators=20,      # Reduzido ainda mais para maior velocidade
                max_depth=5,          # Profundidade limitada
                max_features='sqrt',   # Menos features por árvore
                random_state=42
            )
            self.model.fit(X, y)
            
            # Calcular importância das features
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Selecionar apenas top 5 features para análise
            top_features = self.feature_importance.head(5)['feature'].tolist()
            X_top = X[top_features]
            
            if not self.fast_mode:
                # Calcular SHAP values apenas se não estiver em modo rápido
                try:
                    explainer = shap.TreeExplainer(self.model)
                    self.shap_values = explainer.shap_values(X_top)
                except Exception as e:
                    print(f"Aviso: Não foi possível calcular valores SHAP: {str(e)}")
                    self.shap_values = None
            
            return self
            
        except Exception as e:
            print(f"Erro ao treinar modelo: {str(e)}")
            return None
    
    def plot_causal_analysis(self, output_dir='reports/causal_analysis'):
        """Plota análise causal"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            if self.model is None:
                self.fit_model()
            
            if self.model is not None:
                # 1. Importância das Features (Top 5)
                self._plot_feature_importance(output_dir)
                
                # 2. SHAP Summary (apenas se não estiver em modo rápido)
                if not self.fast_mode and self.shap_values is not None:
                    self._plot_shap_summary(output_dir)
                else:
                    self._plot_feature_correlation(output_dir)
            
        except Exception as e:
            print(f"Erro ao gerar visualizações: {str(e)}")
    
    def _plot_feature_importance(self, output_dir):
        """Plota importância das features"""
        # Plotar apenas top 5 features
        top_features = self.feature_importance.head(5)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h'
            )
        ])
        
        fig.update_layout(
            title='Top 5 Features Mais Importantes para Churn',
            xaxis_title='Importância',
            yaxis_title='Feature',
            height=400
        )
        
        fig.write_html(f'{output_dir}/feature_importance.html')
    
    def _plot_feature_correlation(self, output_dir):
        """Plota correlação entre features e churn (alternativa rápida ao SHAP)"""
        # Calcular correlações com o target
        correlations = []
        top_features = self.feature_importance.head(5)['feature'].tolist()
        
        for feature in top_features:
            corr = self.data[feature].corr(self.data[self.target])
            correlations.append({
                'feature': feature,
                'correlation': abs(corr)
            })
        
        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=corr_df['correlation'],
                y=corr_df['feature'],
                orientation='h'
            )
        ])
        
        fig.update_layout(
            title='Correlação das Top Features com Churn',
            xaxis_title='Correlação Absoluta',
            yaxis_title='Feature',
            height=400
        )
        
        fig.write_html(f'{output_dir}/feature_correlation.html')
    
    def _plot_shap_summary(self, output_dir):
        """Plota SHAP summary"""
        if self.shap_values is None:
            return
            
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            self.data[self.feature_importance['feature'].head(5)],
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_dependence_analysis(self, output_dir):
        """Plota análise de dependência"""
        # Criar subplots para as top 3 features
        fig = make_subplots(rows=1, cols=3,
                           subplot_titles=self.feature_importance['feature'][:3])
        
        X, _ = self.prepare_data()
        for i, feature in enumerate(self.feature_importance['feature'][:3], 1):
            fig.add_trace(
                go.Scatter(
                    x=X[feature],
                    y=self.shap_values[:, X.columns.get_loc(feature)],
                    mode='markers',
                    name=feature
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            height=400,
            title_text="Análise de Dependência (Top 3 Features)"
        )
        
        fig.write_html(f'{output_dir}/dependence_analysis.html')
    
    def _plot_interaction_effects(self, output_dir):
        """Plota efeitos de interação entre features"""
        if self.shap_values is None:
            self.fit_model()
        
        # Pegar top features
        top_features = self.feature_importance['feature'].head(5).tolist()
        X = self.data.drop(columns=[self.target])
        
        # Criar subplots para cada par de features
        fig = make_subplots(rows=2, cols=2)
        
        for i, feat1 in enumerate(top_features[:2]):
            for j, feat2 in enumerate(top_features[2:4]):
                shap_interaction = shap.dependence_plot(
                    feat1, self.shap_values, X,
                    interaction_index=feat2,
                    show=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=X[feat1],
                        y=self.shap_values[:, X.columns.get_loc(feat1)],
                        mode='markers',
                        marker=dict(
                            color=X[feat2],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name=f'{feat1} vs {feat2}'
                    ),
                    row=i+1, col=j+1
                )
        
        fig.update_layout(height=800, title_text="Efeitos de Interação")
        fig.write_html(f'{output_dir}/interaction_effects.html')
    
    def _plot_causal_paths(self, output_dir):
        """Plota caminhos causais para churn"""
        if self.feature_importance is None:
            self.analyze_feature_importance()
        
        # Criar grafo de caminhos causais
        import networkx as nx
        G = nx.DiGraph()
        
        # Adicionar nós
        top_features = self.feature_importance.head(10)
        for _, row in top_features.iterrows():
            G.add_node(row['feature'], weight=row['importance'])
        
        # Adicionar arestas baseadas em correlações
        X = self.data.drop(columns=[self.target])
        corr_matrix = X[top_features['feature']].corr()
        
        for i, feat1 in enumerate(corr_matrix.index):
            for j, feat2 in enumerate(corr_matrix.columns):
                if i < j and abs(corr_matrix.loc[feat1, feat2]) > 0.3:
                    G.add_edge(feat1, feat2,
                             weight=abs(corr_matrix.loc[feat1, feat2]))
        
        # Criar visualização do grafo
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar arestas
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
        )
        
        # Adicionar nós
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G.nodes[node]['weight'] * 1000)
        
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color='#1f77b4',
                    line_width=2
                ),
                text=node_text,
                textposition='top center',
                hoverinfo='text'
            )
        )
        
        fig.update_layout(
            title='Grafo de Caminhos Causais',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            height=600
        )
        
        fig.write_html(f'{output_dir}/causal_paths.html') 