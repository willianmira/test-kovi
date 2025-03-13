# Dashboard de Análise de Retenção de Motoristas

## 📊 Sobre o Projeto
Dashboard interativo para análise de retenção de motoristas, focando em métricas chave como taxa de evasão, impacto financeiro e segmentação de usuários.

## 🏗️ Estrutura do Projeto
```
.
├── src/
│   └── visualization/
│       └── dashboard.py    # Implementação principal do dashboard
├── requirements.txt        # Dependências do projeto
├── main.py                # Arquivo principal para execução do dashboard
└── README.md              # Este arquivo
```

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- pip (gerenciador de pacotes Python)

### Instalação
1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_DIRETORIO]
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Execução
1. Execute o dashboard:
```bash
python main.py
```

2. Acesse o dashboard no navegador:
```
http://localhost:8050
```

## 📦 Dependências Principais
- dash==2.14.2
- plotly==5.18.0
- pandas==2.1.4
- numpy==1.26.2
- dash-core-components==2.0.0
- dash-html-components==2.0.0
- dash-table==5.0.0

## 📊 Funcionalidades Principais

### 1. Análise de Retenção
- Taxa de evasão mensal
- Tempo médio até evasão
- Impacto financeiro da evasão

### 2. Segmentação de Motoristas
- Distribuição por segmento
- Análise de receita por segmento
- Perfil de comportamento

### 3. Indicadores de Risco
- Padrões de comportamento
- Indicadores preditivos de evasão
- Análise de tendências

### 4. Recomendações Estratégicas
- Priorização de segmentos
- Oportunidades de recuperação
- Análise de impacto financeiro
