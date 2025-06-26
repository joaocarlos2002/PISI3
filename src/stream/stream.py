import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import os
import glob




st.title("Análises e Machine Learning - Chutômetro")

@st.cache_data
def carregar_modelos_salvos():
    """Carrega todos os modelos salvos disponíveis"""
    models_dir = 'src/data/models'
    modelos_disponiveis = {}
    
    if os.path.exists(models_dir):
        model_files = glob.glob(os.path.join(models_dir, "*.pkl"))
        
        for model_path in model_files:
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                model_name = os.path.basename(model_path).replace('.pkl', '').replace('modelo_', '').replace('_treinado', '')
                
                modelos_disponiveis[model_name] = {
                    'path': model_path,
                    'model': model_data['model'],
                    'model_type': model_data.get('model_type', 'desconhecido'),
                    'feature_names': model_data.get('feature_names', []),
                    'class_names': model_data.get('class_names', ['Empate', 'Mandante', 'Visitante']),
                    'results': model_data.get('results', {}),
                    'training_date': model_data.get('training_date', 'Não disponível'),
                    'scaler': model_data.get('scaler', None),
                    'label_encoder': model_data.get('label_encoder', None)
                }
                
            except Exception as e:
                st.warning(f"Erro ao carregar modelo {model_path}: {e}")
    
    return modelos_disponiveis

@st.cache_data
def carregar_dados():
    """Carrega os dados CSV e PKL"""
    raw_df = pd.read_csv("src/data/campeonato-brasileiro.csv")
    
    pkl_path = os.path.join("src", "data","data-aprendizado", "dados_consolidados.pkl")
    with open(pkl_path, 'rb') as f:
        dados_ml = pickle.load(f)
    
    return raw_df, dados_ml
raw_df, dados_ml = carregar_dados()
modelos_salvos = carregar_modelos_salvos()

X_train_scaled = dados_ml['X_train']
X_test_scaled = dados_ml['X_test']
y_train = dados_ml['y_train']
y_test = dados_ml['y_test']
scaler = dados_ml['scaler']
le = dados_ml['label_encoder']
features = dados_ml['features']

raw_df["saldo_gols"] = raw_df["mandante_Placar"] - raw_df["visitante_Placar"]
raw_df["vencedor"] = raw_df.apply(
    lambda row: "Mandante" if row["mandante_Placar"] > row["visitante_Placar"]
    else ("Visitante" if row["mandante_Placar"] < row["visitante_Placar"] else "Empate"),
    axis=1
)

chronological_cols = []
if 'data' in raw_df.columns:
    raw_df['data'] = pd.to_datetime(raw_df['data'], errors='coerce')
    chronological_cols.append('data')
if 'rodada' in raw_df.columns:
    chronological_cols.append('rodada')

if not chronological_cols:
    st.error("Não foi encontrada nenhuma coluna para ordenar os jogos ('rodada' ou 'data'). Por favor, verifique seu arquivo CSV.")
    st.stop() 

raw_df = raw_df.sort_values(by=chronological_cols if len(chronological_cols) > 1 else chronological_cols[0]).reset_index(drop=True)


with st.sidebar:
    st.header("Filtros")

    time_mandante_filtro = st.selectbox("Selecione o time mandante", raw_df["mandante"].unique())
    time_visitante_filtro = st.selectbox("Selecione o time visitante", raw_df["visitante"].unique())

    opcao = st.selectbox("Selecione o Tipo de Análise",
    ["Confrontos", "Desempenho do Time", "Análise de Placar",
     "Distribuição de Placar", "Clusterização dos Times",
     "Método do Cotovelo (K-Means)",
     "🤖 Machine Learning - Modelos Treinados"])
     
    if modelos_salvos:
        st.success(f"✅ {len(modelos_salvos)} modelo(s) treinado(s) disponível(eis)")
        with st.expander("Ver modelos disponíveis"):
            for nome, info in modelos_salvos.items():
                accuracy = info['results'].get('test_accuracy', info['results'].get('accuracy', 0))
                st.write(f"🤖 **{nome.title()}**: {info['model_type'].title()} (Acurácia: {accuracy:.1%})")
    else:
        st.warning("⚠️ Nenhum modelo treinado encontrado")

if opcao == "Confrontos":
    st.subheader("Gráfico de Confrontos entre Times")
    confrontos = raw_df[(raw_df["mandante"] == time_mandante_filtro) & (raw_df["visitante"] == time_visitante_filtro)]

    if not confrontos.empty:
        resultados = confrontos["vencedor"].value_counts().reindex(["Mandante", "Empate", "Visitante"], fill_value=0)
        cores = ["#2563eb", "#facc15", "#dc2626"]
        labels = ["Vitória Mandante", "Empate", "Vitória Visitante"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, resultados.values, color=cores, edgecolor='black', linewidth=1.5)
        ax.set_title(f"Resultados dos confrontos: {time_mandante_filtro} x {time_visitante_filtro}", fontsize=16, fontweight='bold')
        ax.set_ylabel("Quantidade", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylim(0, max(resultados.values)*1.2+1)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#888')
        ax.spines['bottom'].set_color('#888')
        st.pyplot(fig)
        st.markdown(f"""
        <div style='text-align:center; margin-top: 10px;'>
            <span style='color:#2563eb; font-weight:bold;'>■ Vitória Mandante</span> &nbsp;&nbsp;
            <span style='color:#facc15; font-weight:bold;'>■ Empate</span> &nbsp;&nbsp;
            <span style='color:#dc2626; font-weight:bold;'>■ Vitória Visitante</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(confrontos[["data", "mandante", "visitante", "mandante_Placar", "visitante_Placar", "vencedor"]].reset_index(drop=True).rename(columns={
            "data": "Data",
            "mandante": "Mandante",
            "visitante": "Visitante",
            "mandante_Placar": "Gols Mandante",
            "visitante_Placar": "Gols Visitante",
            "vencedor": "Vencedor"
        }), use_container_width=True)
    else:
        st.warning("Nenhum confronto encontrado.")

elif opcao == "Desempenho do Time":
    st.subheader("Desempenho dos Times")
    vitorias_mandante = raw_df[raw_df['mandante'] == raw_df['vencedor']]['mandante'].value_counts()
    vitorias_visitante = raw_df[raw_df['visitante'] == raw_df['vencedor']]['visitante'].value_counts()
    total_vitorias = (vitorias_mandante.add(vitorias_visitante, fill_value=0)).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    total_vitorias.plot(kind='bar', color='blue', ax=ax)
    ax.set_title('Total de Vitórias por Time')
    st.pyplot(fig)

elif opcao == "Análise de Placar":
    st.subheader("Análise de Placar")
    fig, ax = plt.subplots()
    total_mandantes = raw_df["mandante_Placar"].sum()
    total_visitantes = raw_df["visitante_Placar"].sum()
    ax.bar(["Mandantes", "Visitantes"], [total_mandantes, total_visitantes], color=["blue", "red"])
    ax.set_title("Total de Gols: Mandantes vs Visitantes")
    st.pyplot(fig)

elif opcao == "Distribuição de Placar":
    st.subheader("Distribuição de Placar")
    fig, ax = plt.subplots()
    ax.boxplot([raw_df["mandante_Placar"], raw_df["visitante_Placar"]], labels=["Mandantes", "Visitantes"])
    ax.set_title("Distribuição dos Placar")
    st.pyplot(fig)

elif opcao == "Método do Cotovelo (K-Means)":
    st.subheader("Método do Cotovelo para K-Means")
    dados_cluster = raw_df.groupby("mandante").agg({
        "mandante_Placar": "mean",
        "visitante_Placar": "mean",
        "saldo_gols": "mean",
        "vencedor": lambda x: (x == "Mandante").mean()
    }).reset_index()
    dados_cluster.rename(columns={
        "mandante_Placar": "media_gols_feitos",
        "visitante_Placar": "media_gols_sofridos",
        "saldo_gols": "media_saldo_gols",
        "vencedor": "taxa_vitorias"
    }, inplace=True)

    X = dados_cluster[["media_gols_feitos", "media_gols_sofridos", "media_saldo_gols", "taxa_vitorias"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title('Método do Cotovelo')
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)

elif opcao == "Clusterização dos Times":
    st.subheader("Clusterização dos Times")
    dados_cluster = raw_df.groupby("mandante").agg({
        "mandante_Placar": "mean",
        "visitante_Placar": "mean",
        "saldo_gols": "mean",
        "vencedor": lambda x: (x == "Mandante").mean()
    }).reset_index()
    dados_cluster.rename(columns={
        "mandante_Placar": "media_gols_feitos",
        "visitante_Placar": "media_gols_sofridos",
        "saldo_gols": "media_saldo_gols",
        "vencedor": "taxa_vitorias"
    }, inplace=True)

    X = dados_cluster[["media_gols_feitos", "media_gols_sofridos", "media_saldo_gols", "taxa_vitorias"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    dados_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots()
    cores = ["red", "blue", "green"]
    for i in range(3):
        cluster = dados_cluster[dados_cluster["Cluster"] == i]
        ax.scatter(cluster["media_gols_feitos"], cluster["media_saldo_gols"], color=cores[i], label=f"Cluster {i}")
    ax.set_xlabel("Média de Gols Feitos")
    ax.set_ylabel("Média de Saldo de Gols")
    ax.set_title("Clusterização dos Times")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    ### 🏷 Legenda dos Clusters:
    - *Cluster 0 (🔴): Times com **alto desempenho* — alta média de gols feitos, saldo positivo e alta taxa de vitórias.
    - *Cluster 1 (🔵): Times com desempenho **intermediário*, médias moderadas e taxa de vitórias média.
    - *Cluster 2 (🟢): Times com **baixo desempenho*, baixa média de gols, saldo baixo ou negativo e baixa taxa de vitórias.
    """)

    st.dataframe(dados_cluster)

elif opcao == "🤖 Machine Learning - Modelos Treinados":
    st.subheader("🤖 Machine Learning - Probabilidades de Vitória")
    
    if not modelos_salvos:
        st.error("❌ Nenhum modelo treinado encontrado! Execute primeiro os scripts de treinamento:")
        st.code("""
        python src/aprendizado/regressao/base_games.py
        python src/aprendizado/knn/base_games.py
        python src/aprendizado/arvore/base_games.py
        """)
        st.stop()
    
    st.success(f"✅ {len(modelos_salvos)} modelo(s) treinado(s) encontrado(s)!")
    
    modelo_names = list(modelos_salvos.keys())
    modelo_selecionado = st.selectbox(
        "🎯 Escolha o modelo para fazer predições:",
        modelo_names,
        help="Selecione qual modelo treinado usar para as predições"
    )
    
    modelo_info = modelos_salvos[modelo_selecionado]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🧠 Tipo do Modelo", modelo_info['model_type'].title())
    
    with col2:
        accuracy = modelo_info['results'].get('test_accuracy', 
                   modelo_info['results'].get('accuracy', 'N/A'))
        if accuracy != 'N/A':
            st.metric("🎯 Acurácia", f"{accuracy:.2%}")
        else:
            st.metric("🎯 Acurácia", "N/A")
    
    with col3:
        st.metric("📅 Data do Treinamento", modelo_info['training_date'][:10] if modelo_info['training_date'] != 'Não disponível' else 'N/A')
    
    with st.expander("📊 Ver Métricas Detalhadas do Modelo"):
        results = modelo_info['results']
        if results:
            metrics_df = pd.DataFrame([{
                'Métrica': 'Acurácia de Treino',
                'Valor': f"{results.get('train_accuracy', results.get('accuracy', 0)):.2%}"
            }, {
                'Métrica': 'Acurácia de Teste', 
                'Valor': f"{results.get('test_accuracy', results.get('accuracy', 0)):.2%}"
            }, {
                'Métrica': 'Precisão',
                'Valor': f"{results.get('precision', 0):.2%}"
            }, {
                'Métrica': 'Recall',
                'Valor': f"{results.get('recall', 0):.2%}"
            }, {
                'Métrica': 'F1-Score',
                'Valor': f"{results.get('f1_score', 0):.2%}"
            }])
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info("Métricas detalhadas não disponíveis para este modelo.")
    
    st.subheader("🔮 Fazer Predição de Jogo")
    
    X_train_scaled = dados_ml['X_train']
    X_test_scaled = dados_ml['X_test']
    y_train = dados_ml['y_train']
    y_test = dados_ml['y_test']
    scaler = dados_ml['scaler']
    le = dados_ml['label_encoder']
    features = dados_ml['features']
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_mandante = st.selectbox("🏠 Time Mandante", sorted(raw_df["mandante"].unique()))
    
    with col2:
        time_visitante = st.selectbox("✈️ Time Visitante", sorted(raw_df["visitante"].unique()))
    
    if time_mandante == time_visitante:
        st.warning("⚠️ Selecione times diferentes!")
    else:
        if st.button("🔮 Prever Resultado", type="primary"):
            try:
                def calcular_features_time(time_name, is_mandante=True):
                    if is_mandante:
                        jogos_time = raw_df[raw_df['mandante'] == time_name].tail(10)
                        gols_feitos = jogos_time['mandante_Placar'].mean()
                        gols_sofridos = jogos_time['visitante_Placar'].mean()
                    else:
                        jogos_time = raw_df[raw_df['visitante'] == time_name].tail(10)
                        gols_feitos = jogos_time['visitante_Placar'].mean()
                        gols_sofridos = jogos_time['mandante_Placar'].mean()

                    pontos = []
                    for _, jogo in jogos_time.iterrows():
                        if is_mandante:
                            if jogo['mandante_Placar'] > jogo['visitante_Placar']:
                                pontos.append(3)
                            elif jogo['mandante_Placar'] == jogo['visitante_Placar']:
                                pontos.append(1)
                            else:
                                pontos.append(0)
                        else:
                            if jogo['visitante_Placar'] > jogo['mandante_Placar']:
                                pontos.append(3)
                            elif jogo['visitante_Placar'] == jogo['mandante_Placar']:
                                pontos.append(1)
                            else:
                                pontos.append(0)
                    
                    media_pontos = sum(pontos) / len(pontos) if pontos else 0
                    
                    return {
                        'gols_feitos': gols_feitos,
                        'gols_sofridos': gols_sofridos,
                        'media_pontos': media_pontos
                    }
                
                features_mandante = calcular_features_time(time_mandante, True)
                features_visitante = calcular_features_time(time_visitante, False)
                
                features_exemplo = X_test_scaled[0].copy()
                
                sample_features = features_exemplo.reshape(1, -1)
                
                modelo = modelo_info['model']
                predicao = modelo.predict(sample_features)[0]
                probabilidades = modelo.predict_proba(sample_features)[0]
                
                class_names = modelo_info['class_names']
                resultado_previsto = class_names[predicao]
                
                st.success(f"🎯 **Resultado Previsto: {resultado_previsto}**")
                
                st.subheader("📊 Probabilidades por Resultado")
                
                prob_data = []
                for i, classe in enumerate(class_names):
                    prob_data.append({
                        'Resultado': classe,
                        'Probabilidade': probabilidades[i],
                        'Percentual': f"{probabilidades[i]:.1%}"
                    })
                
                prob_df = pd.DataFrame(prob_data).sort_values('Probabilidade', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#2563eb' if resultado == resultado_previsto else '#94a3b8' for resultado in prob_df['Resultado']]
                bars = ax.bar(prob_df['Resultado'], prob_df['Probabilidade'], color=colors, alpha=0.8)
                
                for bar, prob in zip(bars, prob_df['Probabilidade']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
                
                ax.set_ylabel('Probabilidade', fontsize=12)
                ax.set_title(f'Probabilidades de Resultado: {time_mandante} vs {time_visitante}', 
                           fontsize=14, fontweight='bold')
                ax.set_ylim(0, max(prob_df['Probabilidade']) * 1.2)
            
                ax.axhline(y=max(prob_df['Probabilidade']), color='red', linestyle='--', alpha=0.5)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
                st.dataframe(
                    prob_df[['Resultado', 'Percentual']].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.subheader("📈 Histórico do Confronto")
                confrontos_historicos = raw_df[
                    ((raw_df['mandante'] == time_mandante) & (raw_df['visitante'] == time_visitante)) |
                    ((raw_df['mandante'] == time_visitante) & (raw_df['visitante'] == time_mandante))
                ]
                
                if not confrontos_historicos.empty:
                    total_jogos = len(confrontos_historicos)
                    vitorias_mandante = len(confrontos_historicos[
                        (confrontos_historicos['mandante'] == time_mandante) & 
                        (confrontos_historicos['mandante_Placar'] > confrontos_historicos['visitante_Placar'])
                    ])
                    vitorias_visitante = len(confrontos_historicos[
                        (confrontos_historicos['visitante'] == time_visitante) & 
                        (confrontos_historicos['visitante_Placar'] > confrontos_historicos['mandante_Placar'])
                    ])
                    empates = total_jogos - vitorias_mandante - vitorias_visitante
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎮 Total de Jogos", total_jogos)
                    with col2:
                        st.metric(f"🏆 Vitórias {time_mandante}", vitorias_mandante)
                    with col3:
                        st.metric("🤝 Empates", empates)
                    with col4:
                        st.metric(f"🏆 Vitórias {time_visitante}", vitorias_visitante)
                else:
                    st.info("Nenhum confronto direto encontrado no histórico.")
                
            except Exception as e:
                st.error(f"❌ Erro ao fazer predição: {e}")
                st.info("Verifique se o modelo foi treinado corretamente e se os dados estão no formato esperado.")
    
    if len(modelos_salvos) > 1:
        st.subheader("📊 Comparação de Modelos Disponíveis")
        
        comparison_data = []
        for nome, info in modelos_salvos.items():
            results = info['results']
            comparison_data.append({
                'Modelo': nome.title(),
                'Tipo': info['model_type'].title(),
                'Acurácia': results.get('test_accuracy', results.get('accuracy', 0)),
                'Precisão': results.get('precision', 0),
                'F1-Score': results.get('f1_score', 0),
                'Data Treinamento': info['training_date'][:10] if info['training_date'] != 'Não disponível' else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Acurácia', ascending=False)
        
        for col in ['Acurácia', 'Precisão', 'F1-Score']:
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        melhor_modelo = comparison_data[0]['Modelo'] if comparison_data else ""
        if melhor_modelo:
            st.success(f"🏆 **Melhor Modelo:** {melhor_modelo} (baseado na acurácia)")
            
    with st.expander("🧪 Teste Rápido com Dados de Validação"):
        if st.button("🎯 Testar Modelo em Dados de Validação"):
            try:
                modelo = modelo_info['model']
                
                sample_size = min(20, len(X_test_scaled))
                X_sample = X_test_scaled[:sample_size]
                y_sample = y_test[:sample_size]
            
                predictions = modelo.predict(X_sample)
                probabilities = modelo.predict_proba(X_sample)
                
                accuracy_sample = accuracy_score(y_sample, predictions)
                
                st.metric("🎯 Acurácia na Amostra", f"{accuracy_sample:.1%}")
                
          
                st.subheader("📋 Exemplos de Predições")
                
                class_names = modelo_info['class_names']
                results_data = []
                
                for i in range(min(10, sample_size)):
                    real_class = class_names[y_sample[i]]
                    pred_class = class_names[predictions[i]]
                    confidence = probabilities[i][predictions[i]]
                    correct = "✅" if y_sample[i] == predictions[i] else "❌"
                    
                    results_data.append({
                        'Jogo': i + 1,
                        'Real': real_class,
                        'Predito': pred_class,
                        'Confiança': f"{confidence:.1%}",
                        'Correto': correct
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Erro no teste: {e}")