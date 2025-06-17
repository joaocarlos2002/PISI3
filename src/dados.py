# chutometro_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Título do aplicativo
st.title("Análises e Machine Learning - Chutômetro")

# Carregar os dados
df = pd.read_csv("campeonato-brasileiro.csv")

# Pré-processamento básico
df["saldo_gols"] = df["mandante_Placar"] - df["visitante_Placar"]
df["vencedor"] = df.apply(
    lambda row: "Mandante" if row["mandante_Placar"] > row["visitante_Placar"]
    else ("Visitante" if row["mandante_Placar"] < row["visitante_Placar"] else "Empate"),
    axis=1
)

# ==================== Sidebar ====================
with st.sidebar:
    st.header("Filtros")

    # Filtro de Confronto
    st.subheader("Filtro de Confrontos")
    time_mandante = st.selectbox("Selecione o time mandante", df["mandante"].unique())
    time_visitante = st.selectbox("Selecione o time visitante", df["visitante"].unique())

    # Filtro de Tipo de Análise
    st.subheader("Filtro de Gráfico/Análise")
    opcao = st.selectbox("Selecione o Tipo de Análise", 
                          ["Confrontos", "Desempenho do Time", "Análise de Placar", 
                           "Distribuição de Placar", "Clusterização dos Times", 
                           "Machine Learning - Classificação"])

# ==================== Gráficos ====================
if opcao == "Confrontos":
    st.subheader("Gráfico de Confrontos")
    confrontos = df[(df["mandante"] == time_mandante) & (df["visitante"] == time_visitante)]

    if not confrontos.empty:
        resultados = confrontos["vencedor"].value_counts().reindex(["Mandante", "Empate", "Visitante"], fill_value=0)
        cores = ["blue", "yellow", "red"]

        fig, ax = plt.subplots()
        resultados.plot(kind='bar', color=cores, ax=ax)
        ax.set_title(f"Vitórias entre {time_mandante} e {time_visitante}")
        st.pyplot(fig)
    else:
        st.warning("Nenhum confronto encontrado.")

elif opcao == "Desempenho do Time":
    st.subheader("Desempenho dos Times")
    vitorias_mandante = df[df['mandante'] == df['vencedor']]['mandante'].value_counts()
    vitorias_visitante = df[df['visitante'] == df['vencedor']]['visitante'].value_counts()
    total_vitorias = (vitorias_mandante.add(vitorias_visitante, fill_value=0)).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    total_vitorias.plot(kind='bar', color='blue', ax=ax)
    ax.set_title('Total de Vitórias por Time')
    st.pyplot(fig)

elif opcao == "Análise de Placar":
    st.subheader("Análise de Placar")

    total_mandantes = df["mandante_Placar"].sum()
    total_visitantes = df["visitante_Placar"].sum()

    plt.bar(["Mandantes", "Visitantes"], [total_mandantes, total_visitantes], color=["blue", "red"])
    plt.title("Total de Gols: Mandantes vs Visitantes")
    st.pyplot(plt)

elif opcao == "Distribuição de Placar":
    st.subheader("Distribuição de Placar")
    fig, ax = plt.subplots()
    ax.boxplot([df["mandante_Placar"], df["visitante_Placar"]], labels=["Mandantes", "Visitantes"])
    ax.set_title("Distribuição dos Placar")
    st.pyplot(fig)

    # ==================== Clusterização ====================
elif opcao == "Clusterização dos Times":
    st.subheader("Clusterização dos Times")

    dados_cluster = df.groupby("mandante").agg({
        "mandante_Placar": "sum",
        "visitante_Placar": "sum",
        "saldo_gols": "sum",
        "vencedor": lambda x: (x == "Mandante").sum()
    }).reset_index()

    dados_cluster.rename(columns={
        "mandante_Placar": "gols_feitos",
        "visitante_Placar": "gols_sofridos",
        "vitorias": "vitorias"
    }, inplace=True)

    X = dados_cluster[["gols_feitos", "gols_sofridos", "saldo_gols", "vencedor"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    dados_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots()
    cores = ["red", "blue", "green"]
    for i in range(3):
        cluster = dados_cluster[dados_cluster["Cluster"] == i]
        ax.scatter(cluster["gols_feitos"], cluster["saldo_gols"], color=cores[i], label=f"Cluster {i}")
    ax.set_xlabel("Gols Feitos")
    ax.set_ylabel("Saldo de Gols")
    ax.set_title("Clusterização dos Times")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Dados dos Clusters")
    st.dataframe(dados_cluster)