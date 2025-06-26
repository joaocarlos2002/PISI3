# 📦 GUIA DE MODELOS SALVOS EM .PKL

## 🎯 Funcionalidades Adicionadas

### ✅ Todos os modelos agora salvam automaticamente:
- **Regressão Logística**: `src/data/models/modelo_regressao_treinado.pkl`
- **KNN**: `src/data/models/modelo_knn_treinado.pkl`
- **Árvore de Decisão**: `src/data/models/modelo_tree_treinado.pkl`
- **Floresta Aleatória**: `src/data/models/modelo_forest_treinado.pkl`

### 📋 Informações salvas em cada modelo:
- ✅ Modelo treinado completo
- ✅ Tipo do modelo
- ✅ Melhores hiperparâmetros encontrados
- ✅ Nomes das features utilizadas
- ✅ Nomes das classes
- ✅ Métricas de performance (acurácia, precisão, recall, F1-score)
- ✅ Data e hora do treinamento
- ✅ Label encoder (para regressão logística)

## 🔧 Métodos Disponíveis

### 💾 Salvamento de Modelos
```python
# Salvar modelo (automático após treinamento)
predictor.save_model()  # Usa nome padrão
predictor.save_model('meu_modelo_personalizado.pkl')  # Nome personalizado
```

### 📂 Carregamento de Modelos
```python
# Carregar modelo salvo
predictor.load_model('src/data/models/modelo_knn_treinado.pkl')
```

### 🔮 Fazer Predições com Modelo Salvo
```python
# Fazer predições em novos dados
predictions, probabilities = predictor.predict_from_saved_model(
    'src/data/models/modelo_regressao_treinado.pkl', 
    X_novos_dados
)
```

### 📋 Verificar Informações do Modelo
```python
# Mostrar informações sem carregar o modelo completo
info = predictor.get_model_info('src/data/models/modelo_tree_treinado.pkl')
```

## 🚀 Como Usar

### 1️⃣ Treinar e Salvar Modelos
```bash
# Executar os scripts - modelos serão salvos automaticamente
python src/aprendizado/regressao/base_games.py
python src/aprendizado/knn/base_games.py
python src/aprendizado/arvore/base_games.py
```

### 2️⃣ Gerenciar Modelos Salvos
```bash
# Usar o sistema de gerenciamento
python gerenciar_modelos.py
```

### 3️⃣ Sistema de Gerenciamento de Modelos
O script `gerenciar_modelos.py` oferece:
- 📂 **Listar modelos salvos**: Vê todos os modelos disponíveis
- 📋 **Informações detalhadas**: Métricas, hiperparâmetros, data de treinamento
- 🧪 **Testar modelos**: Fazer predições de exemplo
- 📊 **Comparar modelos**: Ver qual modelo tem melhor performance

## 📊 Exemplo de Uso Prático

### Carregar e usar modelo KNN:
```python
from src.aprendizado.knn.base_games import KNNGamePredictor

# Criar predictor
predictor = KNNGamePredictor()

# Carregar modelo salvo
predictor.load_model('src/data/models/modelo_knn_treinado.pkl')

# Carregar dados para teste
predictor.load_data()

# Fazer predições
predictions = predictor.model.predict(predictor.X_test[:5])
print("Predições:", predictions)
```

### Comparar modelos diferentes:
```python
# O script gerenciar_modelos.py faz isso automaticamente
# Mostra tabela comparativa com acurácia, precisão e F1-score de todos os modelos
```

## 🎯 Vantagens dos Modelos Salvos

### ✅ **Reutilização**: 
- Não precisa treinar novamente
- Carregamento rápido para produção

### ✅ **Comparação**: 
- Fácil comparação entre diferentes modelos
- Histórico de performance

### ✅ **Deploy**: 
- Modelos prontos para produção
- Informações completas sobre o treinamento

### ✅ **Reprodutibilidade**: 
- Mesmos resultados garantidos
- Rastreabilidade completa

## 📁 Estrutura dos Arquivos Salvos

```
src/data/models/
├── modelo_regressao_treinado.pkl    # Regressão Logística
├── modelo_knn_treinado.pkl          # K-Nearest Neighbors  
├── modelo_tree_treinado.pkl         # Árvore de Decisão
└── modelo_forest_treinado.pkl       # Floresta Aleatória
```

## 🔍 Verificação Rápida

Para ver se os modelos foram salvos corretamente:
```bash
python gerenciar_modelos.py
# Escolha opção 4 para comparar todos os modelos
```

---
**💡 Dica**: Use o script `gerenciar_modelos.py` para uma interface interativa completa de gerenciamento dos seus modelos treinados!
