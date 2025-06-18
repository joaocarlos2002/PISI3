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
     "Método do Cotovelo (K-Means)", # <--- Nova opção adicionada aqui
     "Machine Learning - Classificação",
     "Distribuição dos Erros - Melhor Modelo"]
)

# ==================== Gráficos ====================
if opcao == "Confrontos":
    st.subheader("Gráfico de Confrontos")
    confrontos = df[(df["mandante"] == time_mandante) & (df["visitante"] == time_visitante)]

    if not confrontos.empty:
        resultados = confrontos["vencedor"].value_counts().reindex(["Mandante", "Empate", "Visitante"], fill_value=0)
        cores = ["blue", "yellow", "red"]

        fig, ax = plt.subplots()
        resultados.plot(kind='bar', color=cores, ax=ax)
        ax.set_title("Vitórias entre Mandante e Visitante")
        ax.tick_params(axis='x', rotation=0)
        ax.set_xlabel("")
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

# ==================== Método do Cotovelo ====================
elif opcao == "Método do Cotovelo (K-Means)":
    st.subheader("Método do Cotovelo para K-Means")

    dados_cluster = df.groupby("mandante").agg({
        "mandante_Placar": "sum",
        "visitante_Placar": "sum",
        "saldo_gols": "sum",
        "vencedor": lambda x: (x == "Mandante").sum()
    }).reset_index()

    dados_cluster.rename(columns={
        "mandante_Placar": "gols_feitos",
        "visitante_Placar": "gols_sofridos",
        "vencedor": "vitorias"
    }, inplace=True)

    X = dados_cluster[["gols_feitos", "gols_sofridos", "saldo_gols", "vitorias"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []
    # Testar K de 1 a 10 (ou um range que faça sentido para seus dados)
    max_k = 10 
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_k + 1), wcss, marker='o')
    ax.set_title('Método do Cotovelo')
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('WCSS (Soma dos Quadrados Dentro do Cluster)')
    ax.grid(True)
    st.pyplot(fig)

    st.info("O 'cotovelo' no gráfico indica o número ideal de clusters. É o ponto onde a curva se dobra e a diminuição no WCSS se torna menos acentuada.")

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

    # ==================== Machine Learning ====================
elif opcao == "Machine Learning - Classificação":
    st.subheader("Classificação de Resultados")

    # ====== Novo conjunto de features baseado no histórico ======
    desempenho_mandante = df.groupby("mandante").agg({
        "mandante_Placar": "mean",
        "visitante_Placar": "mean",
        "saldo_gols": "mean",
        "vencedor": lambda x: (x == "Mandante").mean()
    }).rename(columns={
        "mandante_Placar": "media_gols_mandante",
        "visitante_Placar": "media_gols_sofridos_mandante",
        "saldo_gols": "media_saldo_mandante",
        "vencedor": "taxa_vitorias_mandante"
    })

    desempenho_visitante = df.groupby("visitante").agg({
        "visitante_Placar": "mean",
        "mandante_Placar": "mean",
        "saldo_gols": "mean",
        "vencedor": lambda x: (x == "Visitante").mean()
    }).rename(columns={
        "visitante_Placar": "media_gols_visitante",
        "mandante_Placar": "media_gols_sofridos_visitante",
        "saldo_gols": "media_saldo_visitante",
        "vencedor": "taxa_vitorias_visitante"
    })

    df_ml = df.copy()

    df_ml = df_ml.merge(desempenho_mandante, left_on="mandante", right_index=True)
    df_ml = df_ml.merge(desempenho_visitante, left_on="visitante", right_index=True)

    features = [
        "media_gols_mandante", "media_gols_sofridos_mandante", "media_saldo_mandante", "taxa_vitorias_mandante",
        "media_gols_visitante", "media_gols_sofridos_visitante", "media_saldo_visitante", "taxa_vitorias_visitante"
    ]

    X = df_ml[features]
    y = df_ml["vencedor"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    modelos = {
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "Regressão Logística": LogisticRegression(max_iter=1000)
    }

    resultados = {}

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        resultados[nome] = acc

    resultados_df = pd.DataFrame.from_dict(resultados, orient="index", columns=["Acurácia"]).sort_values(by="Acurácia", ascending=False)
    st.dataframe(resultados_df)

    melhor_modelo = resultados_df.index[0]
    st.success(f"O modelo com melhor desempenho foi: {melhor_modelo} com acurácia de {resultados_df.iloc[0,0]:.2f}")

    # Matriz de Confusão
    st.subheader("Matriz de Confusão")
    modelo_final = modelos[melhor_modelo]
    y_pred_final = modelo_final.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_final)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

# ==================== Distribuição dos Erros ====================
elif opcao == "Distribuição dos Erros - Melhor Modelo":
    st.subheader("Distribuição dos Erros na Regressão Logística")

    # Usando as mesmas features e dataset preparados no bloco de Machine Learning
    desempenho_mandante = df.groupby("mandante").agg({
        "mandante_Placar": "mean",
        "visitante_Placar": "mean",
        "saldo_gols": "mean",
        "vencedor": lambda x: (x == "Mandante").mean()
    }).rename(columns={
        "mandante_Placar": "media_gols_mandante",
        "visitante_Placar": "media_gols_sofridos_mandante",
        "saldo_gols": "media_saldo_mandante",
        "vencedor": "taxa_vitorias_mandante"
    })

    desempenho_visitante = df.groupby("visitante").agg({
        "visitante_Placar": "mean",
        "mandante_Placar": "mean",
        "saldo_gols": "mean",
        "vencedor": lambda x: (x == "Visitante").mean()
    }).rename(columns={
        "visitante_Placar": "media_gols_visitante",
        "mandante_Placar": "media_gols_sofridos_visitante",
        "saldo_gols": "media_saldo_visitante",
        "vencedor": "taxa_vitorias_visitante"
    })

    df_ml = df.copy()

    df_ml = df_ml.merge(desempenho_mandante, left_on="mandante", right_index=True)
    df_ml = df_ml.merge(desempenho_visitante, left_on="visitante", right_index=True)

    features = [
        "media_gols_mandante", "media_gols_sofridos_mandante", "media_saldo_mandante", "taxa_vitorias_mandante",
        "media_gols_visitante", "media_gols_sofridos_visitante", "media_saldo_visitante", "taxa_vitorias_visitante"
    ]

    X = df_ml[features]
    y = df_ml["vencedor"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    # Treina apenas a Regressão Logística (melhor modelo)
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    df_erro = pd.DataFrame({
        "Real": le.inverse_transform(y_test),
        "Previsto": le.inverse_transform(y_pred)
    })

    df_erro["Resultado"] = df_erro.apply(
        lambda x: "Acerto" if x["Real"] == x["Previsto"] else "Erro", axis=1
    )

    # Gráfico de distribuição dos erros
    fig, ax = plt.subplots(figsize=(8, 5))
    df_erro.groupby(["Real", "Resultado"]).size().unstack().plot(
        kind="bar", stacked=True, ax=ax, color=["green", "red"]
    )
    ax.set_title("Distribuição dos Erros na Regressão Logística")
    ax.set_ylabel("Quantidade")
    ax.set_xlabel("Classe Real")
    plt.xticks(rotation=0)

    st.pyplot(fig)

    # Mostra tabela resumo
    st.subheader("Tabela de Erros e Acertos")
    st.dataframe(df_erro.groupby(["Real", "Resultado"]).size().unstack().fillna(0))

