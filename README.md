# Dashboard de Análise de Retenção de Motoristas

## 📊 Sobre o Projeto
Dashboard interativo para análise de retenção de motoristas, focando em métricas chave como taxa de evasão, impacto financeiro e segmentação de usuários. O projeto inclui um pipeline completo de análise de churn e um dashboard interativo para visualização dos resultados.

## 🏗️ Estrutura do Projeto
```
.
├── src/
│   ├── visualization/
│   │   └── dashboard.py    # Dashboard interativo
│   ├── models/            # Modelos de machine learning
│   ├── data/             # Processamento de dados
│   ├── analysis/         # Análises e métricas
│   └── main.py          # Pipeline principal
├── data/
│   ├── raw/             # Dados brutos
│   └── processed/       # Dados processados
├── reports/            # Relatórios e visualizações geradas
├── requirements.txt    # Dependências do projeto
├── main.py            # Arquivo de execução
└── README.md          # Este arquivo
```

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- pip (gerenciador de pacotes Python)
- Git

### Instalação
1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_DIRETORIO]
```

2. Crie e ative um ambiente virtual (recomendado):
```bash
# No Windows
python -m venv venv
venv\\Scripts\\activate

# No macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Execução

#### Pipeline Completo (Análise + Dashboard)
Este modo executa todo o processo: processamento de dados, análise de churn, treinamento do modelo e inicialização do dashboard.

1. Certifique-se de que seus dados estão na pasta correta:
```
data/
└── raw/
    └── transactions.csv    # Seus dados de transações
```

2. Execute o pipeline completo:
```bash
python src/main.py
```

O pipeline irá:
1. Processar os dados brutos
2. Calcular métricas de churn
3. Realizar análise de cohort
4. Treinar e avaliar modelos
5. Gerar visualizações
6. Iniciar o dashboard

#### Apenas Dashboard
Se você já tem os dados processados e quer apenas visualizar o dashboard:

```bash
python main.py
```

### Acessando o Dashboard
Após a execução, acesse o dashboard em:
```
http://localhost:8050
```

## 📊 Funcionalidades do Dashboard

### 1. Análise de Retenção
- **Taxa de Evasão Mensal**: Visualização da tendência de churn ao longo do tempo
- **Tempo Médio até Evasão**: Distribuição do tempo até o churn
- **Impacto Financeiro**: Análise do impacto monetário do churn

### 2. Segmentação de Motoristas
- **Distribuição por Segmento**: Visualização da composição da base
- **Análise de Receita**: Receita total e média por segmento
- **Perfil de Comportamento**: Padrões de uso por segmento

### 3. Indicadores de Risco
- **Padrões de Comportamento**: Análise de uso e transações
- **Indicadores Preditivos**: Principais fatores de risco
- **Análise de Tendências**: Evolução temporal dos indicadores

### 4. Recomendações Estratégicas
- **Priorização**: Segmentos prioritários para ação
- **Oportunidades**: Áreas de maior potencial de retenção
- **Impacto**: Análise de ROI das ações

## 📦 Dependências
```
# Visualização
dash==2.14.2
plotly==5.18.0
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-table==5.0.0

# Análise de Dados
pandas==2.1.4
numpy==1.26.2
scikit-learn>=1.2.2
scipy>=1.10.1
```

## 📈 Formato dos Dados
O arquivo de transações (`transactions.csv`) deve conter as seguintes colunas:
- `transaction_date`: Data da transação (formato: YYYY-MM-DD)
- `driver_id`: ID único do motorista
- `amount`: Valor da transação
- `kind`: Tipo de transação


## 📝 Notas
- O dashboard atualiza automaticamente quando novos dados são processados
- As visualizações são interativas e podem ser exportadas em diversos formatos
- Os modelos são retreinados automaticamente quando novos dados são adicionados
