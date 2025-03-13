import pandas as pd
import os
from src.data_preparation import DataPreparation
from src.data.feature_engineering import ChurnFeatureEngineering
from src.models.modeling import ChurnModel
from src.visualization.analytics import ChurnAnalytics
from src.analysis.churn_metrics import ChurnMetrics
from src.analysis.cohort_analysis import CohortAnalysis
from src.analysis.causal_analysis import CausalAnalysis
from src.analysis.price_sensitivity import PriceSensitivityAnalysis
from src.visualization.dashboard import ChurnDashboard
import pickle
from datetime import datetime
from pathlib import Path

def save_processed_data(data, churn_metrics, model_metrics=None):
    """Salva os dados processados e métricas em arquivos"""
    # Criar diretório de dados processados se não existir
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar dados em CSV
    data.to_csv(processed_dir / 'processed_transactions.csv', index=False)
    
    # Salvar métricas em formato pickle
    with open(processed_dir / 'churn_metrics.pkl', 'wb') as f:
        pickle.dump(churn_metrics, f)
    
    if model_metrics:
        with open(processed_dir / 'model_metrics.pkl', 'wb') as f:
            pickle.dump(model_metrics, f)
    
    # Salvar metadata
    metadata = {
        'last_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'rows': len(data),
        'columns': list(data.columns)
    }
    
    with open(processed_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

def load_processed_data():
    """Carrega os dados processados e métricas dos arquivos"""
    processed_dir = Path('data/processed')
    
    if not processed_dir.exists():
        raise FileNotFoundError("Diretório de dados processados não encontrado. Execute a análise primeiro.")
    
    # Carregar dados
    data = pd.read_csv(processed_dir / 'processed_transactions.csv')
    
    # Carregar métricas de churn
    with open(processed_dir / 'churn_metrics.pkl', 'rb') as f:
        churn_metrics = pickle.load(f)
    
    # Tentar carregar métricas do modelo
    model_metrics = None
    if (processed_dir / 'model_metrics.pkl').exists():
        with open(processed_dir / 'model_metrics.pkl', 'rb') as f:
            model_metrics = pickle.load(f)
    
    return data, churn_metrics, model_metrics

def main():
    """Script principal para análise completa de churn"""
    
    # Configurações
    DATA_PATH = "data/raw/transactions.csv"
    TARGET_COLUMN = "churn"
    OUTPUT_DIR = "reports"
    
    # Criar diretórios necessários
    os.makedirs("models", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n=== Iniciando Análise de Churn ===")
    
    # Verificar se já existem dados processados
    try:
        data, churn_metrics, model_metrics = load_processed_data()
        print("Dados processados carregados com sucesso!")
        
        # Iniciar dashboard com dados carregados
        dashboard = ChurnDashboard(data, churn_metrics, model_metrics)
        print("\nIniciando Dashboard...")
        print("Acesse o dashboard em http://localhost:8050")
        dashboard.run_server()
        return
        
    except FileNotFoundError:
        print("Dados processados não encontrados. Iniciando processamento completo...")
    
    # 1. Carregar e Preparar Dados
    print("\n1. Preparação dos Dados")
    print("Carregando dados de data/raw/transactions.csv")
    data_prep = DataPreparation(DATA_PATH)
    
    if data_prep.data is None:
        print("Erro ao carregar os dados. Encerrando...")
        return
    
    # 2. Feature Engineering
    print("\n2. Feature Engineering")
    feature_eng = ChurnFeatureEngineering()
    data_processed = feature_eng.transform(data_prep.data)
    
    # 3. Análise de Métricas de Churn
    print("\n3. Análise de Métricas de Churn")
    churn_metrics = ChurnMetrics(data_processed)
    basic_metrics = churn_metrics.calculate_basic_metrics()
    advanced_metrics = churn_metrics.calculate_advanced_metrics()
    
    print(f"Taxa de Churn: {basic_metrics['churn_rate']:.2%}")
    print(f"Tempo Médio até Churn: {advanced_metrics['avg_days_to_churn']:.1f} dias")
    
    # 4. Análise de Cohort
    print("\n4. Análise de Cohort")
    cohort_analysis = CohortAnalysis(data_processed)
    cohort_analysis.prepare_cohort_data()
    cohort_analysis.plot_cohort_analysis()
    
    # 5. Análise Causal
    print("\n5. Análise Causal")
    causal_analysis = CausalAnalysis(data_processed)
    causal_analysis.plot_causal_analysis()
    
    # 6. Análise de Sensibilidade de Preços
    print("\n6. Análise de Sensibilidade de Preços")
    price_analysis = PriceSensitivityAnalysis(data_processed)
    price_analysis.plot_price_analysis()
    
    # 7. Preparar Features para Modelagem
    print("\n7. Preparando Features para Modelagem")
    X_train, X_test, y_train, y_test, columns = data_prep.prepare_features(
        TARGET_COLUMN
    )
    
    # 8. Treinar e Avaliar Modelo
    print("\n8. Treinando Modelo")
    churn_model = ChurnModel(X_train, X_test, y_train, y_test, columns)
    
    # Treinar com diferentes algoritmos
    print("\nTreinando Random Forest...")
    rf_grid = churn_model.train_model(model_type='rf')
    rf_metrics = churn_model.evaluate_model(rf_grid)
    
    print("\nTreinando Gradient Boosting...")
    gb_grid = churn_model.train_model(model_type='gb')
    gb_metrics = churn_model.evaluate_model(gb_grid)
    
    # Escolher melhor modelo
    best_model = rf_grid if rf_metrics['roc_auc'] > gb_metrics['roc_auc'] else gb_grid
    best_metrics = rf_metrics if rf_metrics['roc_auc'] > gb_metrics['roc_auc'] else gb_metrics
    
    # 9. Gerar Análises do Modelo
    print("\n9. Gerando Análises do Modelo")
    analytics = ChurnAnalytics(
        data=data_processed,
        model=best_model.best_estimator_,
        y_true=y_test,
        y_pred=best_model.predict(X_test),
        y_proba=best_model.predict_proba(X_test)[:, 1]
    )
    
    analytics.generate_eda_plots()
    analytics.generate_model_analysis()
    feature_importance = analytics.generate_feature_importance()
    
    # 10. Salvar Modelo e Dados Processados
    print("\n10. Salvando Modelo e Dados")
    model_path = "models/churn_model.pkl"
    churn_model.save_model(model_path)
    
    # Salvar dados processados
    metrics_dict = {**basic_metrics, **advanced_metrics}
    save_processed_data(data_processed, metrics_dict, best_metrics)
    print("Dados processados salvos com sucesso!")
    
    print("\n=== Análise Completa ===")
    print(f"Todos os resultados foram salvos em '{OUTPUT_DIR}'")
    print(f"Modelo salvo em '{model_path}'")
    
    # 11. Iniciar Dashboard
    print("\n11. Iniciando Dashboard")
    print("Acesse o dashboard em http://localhost:8050")
    dashboard = ChurnDashboard(
        data=data_processed,
        churn_metrics=metrics_dict,
        model_metrics=best_metrics
    )
    dashboard.run_server()

if __name__ == "__main__":
    main() 