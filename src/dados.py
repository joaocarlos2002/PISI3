import lux
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


st.title("Gráficos do aplicativo Chutômetro")
st.write("Dados obtidos através da análise exploratória inicial do dataset, visando responder as seguintes hipóteses:")
st.write("1. O time mandante realmente faz mais gols do que o time visitante por ter sua torcida a favor?")
st.write("2. Quais times mais venceram de todos?")
st.write("3. Há erro de inserção dos dados que podem prejudicar a análise?")


df = pd.read_csv("campeonato-brasileiro.csv")

# Definir o total de vitórias para cada time

vitorias_mandante = df[df['mandante'] == df['vencedor']]['mandante'].value_counts()
vitorias_visitante = df[df['visitante'] == df['vencedor']]['visitante'].value_counts()
total_vitorias = vitorias_mandante.add(vitorias_visitante, fill_value=0).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
total_vitorias.plot(kind='bar', color='skyblue', ax=ax)

ax.set_xlabel('Time')
ax.set_ylabel('Total de Vitórias')
ax.set_title('Total de Vitórias por Time')

# Corrigido: rotacionar e alinhar os rótulos manualmente
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_horizontalalignment('right')

plt.tight_layout()
st.pyplot(fig)


# Gráfico com total de gols mandante vs visitante

total_mandantes = df["mandante_Placar"].sum()
total_visitantes = df["visitante_Placar"].sum()

frequencia_mandante = df["mandante_Placar"].value_counts().sort_index()
frequencia_visitante = df["visitante_Placar"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(["MANDANTES", "VISITANTES"], [total_mandantes, total_visitantes])

plt.title("Total de Gols: Mandantes vs Visitantes")
plt.xlabel("Categoria")
plt.ylabel("Total de Gols")
st.pyplot(plt)
plt.clf()


col1, col2 = st.columns(2)

with col1:
    plt.figure(figsize=(8, 5))
    plt.bar(frequencia_visitante.index, frequencia_visitante.values)

    plt.title("Frequência de Placar dos Visitantes")
    plt.xlabel("Gols do visitante")
    plt.ylabel("Frequência")
    plt.xticks(frequencia_visitante.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)
    plt.clf()

with col2:
    plt.figure(figsize=(8, 5))
    plt.bar(frequencia_mandante.index, frequencia_mandante.values)

    plt.title("Frequência de Placar dos Mandantes")
    plt.xlabel("Gols do mandante")
    plt.ylabel("Frequência")
    plt.xticks(frequencia_mandante.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)
    plt.clf()


# Distribuição em boxplot dos placares - mandantes e visitantes

fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot([df["mandante_Placar"], df["visitante_Placar"]],
           labels=["Mandantes", "Visitantes"],
           patch_artist=True,
           boxprops=dict(facecolor="lightblue"))

ax.set_title("Distribuição dos Placar - Mandantes vs Visitantes")
ax.set_ylabel("Gols")
st.pyplot(fig)



# Filtros do usuário
time_mandante = st.selectbox("Selecione o time mandante", df["mandante"].unique())
time_visitante = st.selectbox("Selecione o time visitante", df["visitante"].unique())


confrontos = df[
    (df["mandante"] == time_mandante) & 
    (df["visitante"] == time_visitante)
].copy()


if not confrontos.empty:
    # Criar coluna com vencedor
    def determinar_vencedor(row):
        if row["mandante_Placar"] > row["visitante_Placar"]:
            return row["mandante"]
        elif row["mandante_Placar"] < row["visitante_Placar"]:
            return row["visitante"]
        else:
            return "Empate"

    confrontos["vencedor"] = confrontos.apply(determinar_vencedor, axis=1)

    # Contar vitórias
    vitorias = confrontos["vencedor"].value_counts()

    # Gráfico de vitórias
    fig, ax = plt.subplots(figsize=(8, 5))
    vitorias.plot(kind='bar', color='lightgreen', ax=ax)
    ax.set_title(f"Vitórias em confrontos entre {time_mandante} e {time_visitante}")
    ax.set_ylabel("Número de Vitórias")
    ax.set_xlabel("Vencedor")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # Colunas que queremos exibir
    colunas_para_exibir = ["mandante", "visitante", "mandante_Placar", "visitante_Placar", "vencedor"]

    # Filtrar apenas as colunas existentes (em caso de erro anterior)
    colunas_validas = [col for col in colunas_para_exibir if col in confrontos.columns]


else:
    st.warning("Nenhum confronto encontrado entre os times selecionados.")