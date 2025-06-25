import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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




st.title("Análises e Machine Learning - Chutômetro")

raw_df = pd.read_csv("src/data/campeonato-brasileiro.csv")

pkl_path = os.path.join("src", "data","data-aprendizado", "dados_consolidados.pkl")
with open(pkl_path, 'rb') as f:
    dados_ml = pickle.load(f)

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
     "Machine Learning - Probabilidades de Vitória"])

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

elif opcao == "Machine Learning - Probabilidades de Vitória":
    st.subheader("Machine Learning - Probabilidades de Vitória")

    N_GAMES_FORM = 10

    df_form = raw_df.copy()

    main_chronological_col = None
    if 'data' in df_form.columns:
        main_chronological_col = 'data'
    elif 'rodada' in df_form.columns:
        main_chronological_col = 'rodada'
    
    if not main_chronological_col:
        st.error("Não foi possível calcular features de forma recente: faltam as colunas 'rodada' ou 'data' no CSV.")
        st.stop()

    df_form['mandante_points'] = df_form.apply(lambda r: 3 if r['vencedor'] == 'Mandante' else (1 if r['vencedor'] == 'Empate' else 0), axis=1)
    df_form['visitante_points'] = df_form.apply(lambda r: 3 if r['vencedor'] == 'Visitante' else (1 if r['vencedor'] == 'Empate' else 0), axis=1)

    team_stats_list = []

    for index, row in df_form.iterrows():
        team_stats_list.append({
            'team': row['mandante'],
            main_chronological_col: row[main_chronological_col],
            'goals_scored': row['mandante_Placar'],
            'goals_conceded': row['visitante_Placar'],
            'points': row['mandante_points']
        })
        team_stats_list.append({
            'team': row['visitante'],
            main_chronological_col: row[main_chronological_col],
            'goals_scored': row['visitante_Placar'],
            'goals_conceded': row['mandante_Placar'],
            'points': row['visitante_points']
        })

    df_team_performances = pd.DataFrame(team_stats_list)

    df_team_performances = df_team_performances.sort_values(by=['team', main_chronological_col]).reset_index(drop=True)

    df_team_performances[f'rolling_goals_scored_{N_GAMES_FORM}'] = df_team_performances.groupby('team')['goals_scored'].transform(lambda x: x.rolling(window=N_GAMES_FORM, min_periods=1).mean().shift(1))
    df_team_performances[f'rolling_goals_conceded_{N_GAMES_FORM}'] = df_team_performances.groupby('team')['goals_conceded'].transform(lambda x: x.rolling(window=N_GAMES_FORM, min_periods=1).mean().shift(1))
    df_team_performances[f'rolling_points_{N_GAMES_FORM}'] = df_team_performances.groupby('team')['points'].transform(lambda x: x.rolling(window=N_GAMES_FORM, min_periods=1).mean().shift(1))

    df_team_performances.fillna(0, inplace=True)

    df_ml = raw_df.copy()

    df_ml = df_ml.merge(
        df_team_performances[['team', main_chronological_col, f'rolling_goals_scored_{N_GAMES_FORM}', f'rolling_goals_conceded_{N_GAMES_FORM}', f'rolling_points_{N_GAMES_FORM}']],
        left_on=['mandante', main_chronological_col],
        right_on=['team', main_chronological_col],
        how='left',
        suffixes=('', '_mandante_form')
    )
    df_ml.drop(columns=['team'], inplace=True)

    df_ml = df_ml.merge(
        df_team_performances[['team', main_chronological_col, f'rolling_goals_scored_{N_GAMES_FORM}', f'rolling_goals_conceded_{N_GAMES_FORM}', f'rolling_points_{N_GAMES_FORM}']],
        left_on=['visitante', main_chronological_col],
        right_on=['team', main_chronological_col],
        how='left',
        suffixes=('', '_visitante_form')
    )
    df_ml.drop(columns=['team'], inplace=True)

    df_ml[f'diff_rolling_goals_scored_{N_GAMES_FORM}'] = df_ml[f'rolling_goals_scored_{N_GAMES_FORM}'] - df_ml[f'rolling_goals_scored_{N_GAMES_FORM}_visitante_form']
    df_ml[f'diff_rolling_goals_conceded_{N_GAMES_FORM}'] = df_ml[f'rolling_goals_conceded_{N_GAMES_FORM}'] - df_ml[f'rolling_goals_conceded_{N_GAMES_FORM}_visitante_form']
    df_ml[f'diff_rolling_points_{N_GAMES_FORM}'] = df_ml[f'rolling_points_{N_GAMES_FORM}'] - df_ml[f'rolling_points_{N_GAMES_FORM}_visitante_form']

    features = [
        f'rolling_goals_scored_{N_GAMES_FORM}', f'rolling_goals_conceded_{N_GAMES_FORM}', f'rolling_points_{N_GAMES_FORM}',
        f'rolling_goals_scored_{N_GAMES_FORM}_visitante_form', f'rolling_goals_conceded_{N_GAMES_FORM}_visitante_form', f'rolling_points_{N_GAMES_FORM}_visitante_form',
        f'diff_rolling_goals_scored_{N_GAMES_FORM}', f'diff_rolling_goals_conceded_{N_GAMES_FORM}', f'diff_rolling_points_{N_GAMES_FORM}'
    ]
    
    df_ml.dropna(subset=features, inplace=True)

    X = df_ml[features]
    y = df_ml["vencedor"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

    st.info(f"Contagem de classes antes do SMOTE no treino: {pd.Series(y_train).value_counts()}")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    st.info(f"Contagem de classes depois do SMOTE no treino: {pd.Series(y_train_res).value_counts()}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    modelos_grid = {
        "Regressão Logística": {
            "model": LogisticRegression(random_state=42, max_iter=5000, class_weight='balanced', solver='liblinear'),
            "params": {
                "C": [0.5, 1, 5, 10, 50]
            }
        }
    }

    resultados = {}
    modelos_treinados = {}

    for nome, config in modelos_grid.items():
        st.info(f"Treinando e ajustando {nome}... Isso pode demorar um pouco devido ao GridSearchCV.")
        grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring='accuracy', n_jobs=1, verbose=1) 
        grid_search.fit(X_train_scaled, y_train_res)
        
        y_pred = grid_search.best_estimator_.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        resultados[nome] = acc
        modelos_treinados[nome] = grid_search.best_estimator_

        st.write(f"Melhores parâmetros para {nome}: {grid_search.best_params_}")
        st.write(f"Melhor acurácia de CV para {nome}: {grid_search.best_score_:.2f}")

    resultados_df = pd.DataFrame.from_dict(resultados, orient="index", columns=["Acurácia"]).sort_values(by="Acurácia", ascending=False)
    st.dataframe(resultados_df)

    melhor_modelo_nome = resultados_df.index[0]
    st.success(f"O melhor modelo é: {melhor_modelo_nome} com acurácia de {resultados_df.iloc[0, 0]:.2f}")

    modelo_final = modelos_treinados[melhor_modelo_nome]
    y_pred_final = modelo_final.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_final, normalize='true')


    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão Normalizada (proporção por classe)")
    st.pyplot(fig)

    st.subheader("Relatório de Classificação (Precision, Recall, F1-score)")
    report = classification_report(y_test, y_pred_final, target_names=le.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report['precision'] = df_report['precision'].apply(lambda x: f"{x:.2%}")
    df_report['recall'] = df_report['recall'].apply(lambda x: f"{x:.2%}")
    df_report['f1-score'] = df_report['f1-score'].apply(lambda x: f"{x:.2%}")
    st.dataframe(df_report)

    st.subheader("Prever um Novo Jogo")

    time1_pred = st.selectbox("Selecione o Time 1 (Mandante)", raw_df["mandante"].unique(), key="time1_pred")
    time2_pred = st.selectbox("Selecione o Time 2 (Visitante)", raw_df["visitante"].unique(), key="time2_pred")

    new_X_scaled = pd.DataFrame(columns=features)

    if st.button("Prever Resultado do Jogo"):
        def get_team_form(team_name, df_team_performances_full, n_games, main_chronological_col):
            team_data = df_team_performances_full[df_team_performances_full['team'] == team_name]
            if team_data.empty:
                st.warning(f"Não há dados de forma recente para o time '{team_name}'. Usando 0 para as features de forma.")
                return {
                    f'rolling_goals_scored_{n_games}': 0,
                    f'rolling_goals_conceded_{n_games}': 0,
                    f'rolling_points_{n_games}': 0
                }
            latest_form = team_data.sort_values(by=main_chronological_col, ascending=False).iloc[0]
            return {
                f'rolling_goals_scored_{n_games}': latest_form[f'rolling_goals_scored_{n_games}'],
                f'rolling_goals_conceded_{n_games}': latest_form[f'rolling_goals_conceded_{n_games}'],
                f'rolling_points_{n_games}': latest_form[f'rolling_points_{n_games}']
            }

        mandante_form = get_team_form(time1_pred, df_team_performances, N_GAMES_FORM, main_chronological_col)
        visitante_form = get_team_form(time2_pred, df_team_performances, N_GAMES_FORM, main_chronological_col)

        new_data = {
            f'rolling_goals_scored_{N_GAMES_FORM}': mandante_form[f'rolling_goals_scored_{N_GAMES_FORM}'],
            f'rolling_goals_conceded_{N_GAMES_FORM}': mandante_form[f'rolling_goals_conceded_{N_GAMES_FORM}'],
            f'rolling_points_{N_GAMES_FORM}': mandante_form[f'rolling_points_{N_GAMES_FORM}'],
            f'rolling_goals_scored_{N_GAMES_FORM}_visitante_form': visitante_form[f'rolling_goals_scored_{N_GAMES_FORM}'],
            f'rolling_goals_conceded_{N_GAMES_FORM}_visitante_form': visitante_form[f'rolling_goals_conceded_{N_GAMES_FORM}'],
            f'rolling_points_{N_GAMES_FORM}_visitante_form': visitante_form[f'rolling_points_{N_GAMES_FORM}'],
        }
        new_data[f'diff_rolling_goals_scored_{N_GAMES_FORM}'] = new_data[f'rolling_goals_scored_{N_GAMES_FORM}'] - new_data[f'rolling_goals_scored_{N_GAMES_FORM}_visitante_form']
        new_data[f'diff_rolling_goals_conceded_{N_GAMES_FORM}'] = new_data[f'rolling_goals_conceded_{N_GAMES_FORM}'] - new_data[f'rolling_goals_conceded_{N_GAMES_FORM}_visitante_form']
        new_data[f'diff_rolling_points_{N_GAMES_FORM}'] = new_data[f'rolling_points_{N_GAMES_FORM}'] - new_data[f'rolling_points_{N_GAMES_FORM}_visitante_form']
        new_X = pd.DataFrame([new_data], columns=features)
        new_X_scaled = scaler.transform(new_X)
        probabilidades = modelo_final.predict_proba(new_X_scaled)[0]
        resultado_previsto_cod = modelo_final.predict(new_X_scaled)[0]
        resultado_previsto_label = le.inverse_transform([resultado_previsto_cod])[0]
        st.subheader(f"Probabilidades de Resultado para {time1_pred} (Mandante) vs {time2_pred} (Visitante):")
        st.write(f"Resultado Previsto: *{resultado_previsto_label}*")
        prob_df = pd.DataFrame({
            "Resultado": le.classes_,
            "Probabilidade": probabilidades
        }).sort_values(by="Probabilidade", ascending=False)
        st.dataframe(prob_df.style.format({'Probabilidade': "{:.2%}"}))
        fig_prob, ax_prob = plt.subplots()
        sns.barplot(x="Resultado", y="Probabilidade", data=prob_df, palette="viridis", ax=ax_prob)
        ax_prob.set_title("Probabilidades do Resultado")
        ax_prob.set_ylim(0, 1)
        st.pyplot(fig_prob)
    y_prob_test_set = modelo_final.predict_proba(X_test_scaled)
    df_prob_test_set = pd.DataFrame(y_prob_test_set, columns=[f"Prob_{cls}" for cls in le.classes_])
    df_result_test_set = pd.DataFrame({'Resultado Real': le.inverse_transform(y_test), 'Resultado Previsto': le.inverse_transform(y_pred_final)})
    df_prob_test_set = pd.concat([df_prob_test_set.reset_index(drop=True), df_result_test_set.reset_index(drop=True)], axis=1)
    st.subheader("Probabilidades Previstas no Conjunto de Teste")
    st.dataframe(df_prob_test_set)