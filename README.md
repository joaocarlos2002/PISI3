# Projeto PISI3

Este repositório contém um projeto de Machine Learning desenvolvido em Python.

## Estrutura de Pastas

```
PISI3/
├── README.md
├── requirements.txt
├── src/
│   ├── aprendizado/
│   │   ├── arvore/
│   │   │   └── base_games.py
│   │   ├── knn/
│   │   │   └── base_games.py
│   │   └── regressao/
│   │       └── base_games.py
│   ├── data/
│   │   ├── campeonato-brasileiro.csv
│   │   ├── data-aprendizado/
│   │   └── figuras/
│   └── pré-processamento/
│       └── pre_processamento.py
└── stream/
```

- `src/`: Código-fonte do projeto.
  - `aprendizado/`: Algoritmos de aprendizado de máquina (árvore de decisão, KNN, regressão logística).
  - `data/`: Dados utilizados e gerados pelo projeto.
    - `figuras/`: Gráficos e imagens geradas.
    - `data-aprendizado/`: Dados processados para aprendizado.
  - `pré-processamento/`: Scripts de pré-processamento dos dados.
  - `stream/`: codigo do stream.
- `requirements.txt`: Dependências do projeto.
- `README.md`: Este arquivo de instruções.

## Como executar o projeto

### 1. Clone o repositório

Abra o terminal e execute:

```bash
git clone https://github.com/joaocarlos2002/PISI3
cd PISI3
```

### 2. Crie e ative um ambiente virtual (venv)

No Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

No Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

Com o ambiente virtual ativado, execute:

```bash
pip install -r requirements.txt
```

