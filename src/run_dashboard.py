"""
Script para executar apenas o dashboard com dados já processados
"""
from main import load_processed_data
from visualization.dashboard import ChurnDashboard

def run_dashboard():
    try:
        # Carregar dados processados
        print("Carregando dados processados...")
        data, churn_metrics, model_metrics = load_processed_data()
        print("Dados carregados com sucesso!")
        
        # Mostrar informações sobre os dados
        print(f"\nInformações dos dados:")
        print(f"- Total de registros: {len(data):,}")
        print(f"- Colunas disponíveis: {', '.join(data.columns)}")
        print(f"- Taxa de Churn: {churn_metrics['churn_rate']:.1%}")
        print(f"- Receita em Risco: R$ {churn_metrics['revenue_at_risk']:,.2f}")
        
        # Iniciar dashboard
        print("\nIniciando Dashboard...")
        print("Acesse o dashboard em http://localhost:8050")
        dashboard = ChurnDashboard(data, churn_metrics, model_metrics)
        dashboard.run_server()
        
    except FileNotFoundError as e:
        print("\nErro: Dados processados não encontrados!")
        print("Execute primeiro o arquivo main.py para processar os dados.")
        print("Comando: python src/main.py")
        
    except Exception as e:
        print(f"\nErro ao carregar dashboard: {str(e)}")
        print("Por favor, verifique se os dados foram processados corretamente.")

if __name__ == "__main__":
    run_dashboard() 