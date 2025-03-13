from src.visualization.dashboard import ChurnDashboard
import pandas as pd
import numpy as np
from datetime import datetime

if __name__ == '__main__':
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