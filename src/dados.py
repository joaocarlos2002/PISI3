import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Gráficos do aplicativo Chutômetro")

df = pd.read_csv("campeonato-brasileiro.csv")

# Criação da barra lateral para os filtros
with st.sidebar:

    st.header("Filtros")

    # Adicionar filtro de confrontro para mandante e visitante
    st.subheader("Filtro de Confrontos")
    time_mandante = st.selectbox("Selecione o time mandante", df["mandante"].unique(), key="mandante_selector")
    time_visitante = st.selectbox("Selecione o time visitante", df["visitante"].unique(), key="visitante_selector")

    # Criação da seleção do tipo de gráfico (segundo filtro)
    st.subheader("Filtro de Gráfico")
    opcao_grafico_principal = st.selectbox("Selecione o Tipo de Gráfico",
                                            ["Confrontos", "Desempenho do Time", "Análise de Placar", "Distribuição de Placar"],
                                            key="grafico_principal_selector")

# Exibição dos gráficos de confronto
if opcao_grafico_principal == "Confrontos":
    st.subheader("Gráfico de Confrontos")
    confrontos = df[(df["mandante"] == time_mandante) & (df["visitante"] == time_visitante)].copy()

    if not confrontos.empty:
        # Definição da lógica para o vencedor
        def determinar_vencedor(row):
            if row["mandante_Placar"] > row["visitante_Placar"]:
                return row["mandante"]
            elif row["mandante_Placar"] < row["visitante_Placar"]:
                return row["visitante"]
            else:
                return "Empate"

        confrontos["vencedor"] = confrontos.apply(determinar_vencedor, axis=1)
        vitorias = confrontos["vencedor"].value_counts()

        # Determinar as cores das barras para facilitar a visualização
        cores = []
        for vencedor in vitorias.index:
            if vencedor == time_mandante:
                cores.append('blue')

            elif vencedor == 'Empate':
                cores.append('yellow')

            else:
                cores.append('red')

        # Elaboração do gráfico de vitórias
        fig, ax = plt.subplots(figsize=(8, 5))
        vitorias.plot(kind='bar', color=cores, ax=ax)
        ax.set_title(f"Vitórias em confrontos entre {time_mandante} e {time_visitante}")
        ax.set_ylabel("Número de Vitórias")
        ax.set_xlabel("Vencedor")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.clf()

    else:
        st.warning("Nenhum confronto encontrado entre os times selecionados.")


# Elaboração do Gráfico de Desempenho do time
elif opcao_grafico_principal == "Desempenho do Time":
    st.subheader("Desempenho do Time")

    # Gráfico total de vitórias para cada time
    vitorias_mandante = df[df['mandante'] == df['vencedor']]['mandante'].value_counts()
    vitorias_visitante = df[df['visitante'] == df['vencedor']]['visitante'].value_counts()
    total_vitorias = vitorias_mandante.add(vitorias_visitante, fill_value=0).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    total_vitorias.plot(kind='bar', color='blue', ax=ax)

    ax.set_xlabel('Time')
    ax.set_ylabel('Total de Vitórias')
    ax.set_title('Total de Vitórias por Time')

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

# Elaboração do gráfico de análise de placar
elif opcao_grafico_principal == "Análise de Placar":
    st.subheader("Análise de Placar")

    # Gráfico com total de gols mandante vs visitante
    total_mandantes = df["mandante_Placar"].sum()
    total_visitantes = df["visitante_Placar"].sum()

    plt.figure(figsize=(6, 4))
    plt.bar(["MANDANTES", "VISITANTES"], [total_mandantes, total_visitantes], color=["blue", "red"])

    plt.title("Total de Gols: Mandantes vs Visitantes")
    plt.xlabel("Categoria")
    plt.ylabel("Total de Gols")
    st.pyplot(plt)
    plt.clf()
