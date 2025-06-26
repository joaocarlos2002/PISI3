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
    st.subheader(f"Desempenho dos Times: {time_mandante_filtro} e {time_visitante_filtro}")
    

    jogos_mandante = raw_df[raw_df['mandante'] == time_mandante_filtro]
    jogos_visitante = raw_df[raw_df['visitante'] == time_visitante_filtro]
    

    stats_mandante = {
        'Time': time_mandante_filtro,
        'Jogos como Mandante': len(jogos_mandante),
        'Vitórias como Mandante': len(jogos_mandante[jogos_mandante['vencedor'] == 'Mandante']),
        'Empates como Mandante': len(jogos_mandante[jogos_mandante['vencedor'] == 'Empate']),
        'Derrotas como Mandante': len(jogos_mandante[jogos_mandante['vencedor'] == 'Visitante']),
        'Gols Feitos como Mandante': jogos_mandante['mandante_Placar'].sum(),
        'Gols Sofridos como Mandante': jogos_mandante['visitante_Placar'].sum()
    }
    
    stats_visitante = {
        'Time': time_visitante_filtro,
        'Jogos como Visitante': len(jogos_visitante),
        'Vitórias como Visitante': len(jogos_visitante[jogos_visitante['vencedor'] == 'Visitante']),
        'Empates como Visitante': len(jogos_visitante[jogos_visitante['vencedor'] == 'Empate']),
        'Derrotas como Visitante': len(jogos_visitante[jogos_visitante['vencedor'] == 'Mandante']),
        'Gols Feitos como Visitante': jogos_visitante['visitante_Placar'].sum(),
        'Gols Sofridos como Visitante': jogos_visitante['mandante_Placar'].sum()
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 {time_mandante_filtro} (Mandante)")
        if len(jogos_mandante) > 0:
            st.metric("🎮 Jogos", int(stats_mandante['Jogos como Mandante']))
            st.metric("🏆 Vitórias", int(stats_mandante['Vitórias como Mandante']))
            st.metric("🤝 Empates", int(stats_mandante['Empates como Mandante']))
            st.metric("❌ Derrotas", int(stats_mandante['Derrotas como Mandante']))
            st.metric("⚽ Gols Feitos", int(stats_mandante['Gols Feitos como Mandante']))
            st.metric("🥅 Gols Sofridos", int(stats_mandante['Gols Sofridos como Mandante']))
            
            if stats_mandante['Jogos como Mandante'] > 0:
                pontos = stats_mandante['Vitórias como Mandante'] * 3 + stats_mandante['Empates como Mandante']
                pontos_possiveis = stats_mandante['Jogos como Mandante'] * 3
                aproveitamento = (pontos / pontos_possiveis) * 100
                st.metric("📈 Aproveitamento", f"{aproveitamento:.1f}%")
        else:
            st.warning("Nenhum jogo como mandante encontrado")
    
    with col2:
        st.subheader(f"📊 {time_visitante_filtro} (Visitante)")
        if len(jogos_visitante) > 0:
            st.metric("🎮 Jogos", int(stats_visitante['Jogos como Visitante']))
            st.metric("🏆 Vitórias", int(stats_visitante['Vitórias como Visitante']))
            st.metric("🤝 Empates", int(stats_visitante['Empates como Visitante']))
            st.metric("❌ Derrotas", int(stats_visitante['Derrotas como Visitante']))
            st.metric("⚽ Gols Feitos", int(stats_visitante['Gols Feitos como Visitante']))
            st.metric("🥅 Gols Sofridos", int(stats_visitante['Gols Sofridos como Visitante']))
            
            if stats_visitante['Jogos como Visitante'] > 0:
                pontos = stats_visitante['Vitórias como Visitante'] * 3 + stats_visitante['Empates como Visitante']
                pontos_possiveis = stats_visitante['Jogos como Visitante'] * 3
                aproveitamento = (pontos / pontos_possiveis) * 100
                st.metric("📈 Aproveitamento", f"{aproveitamento:.1f}%")
        else:
            st.warning("Nenhum jogo como visitante encontrado")
    
    if len(jogos_mandante) > 0 or len(jogos_visitante) > 0:
        st.subheader("📊 Comparação Visual")
        
        categorias = ['Vitórias', 'Empates', 'Derrotas']
        mandante_dados = [
            stats_mandante['Vitórias como Mandante'] if len(jogos_mandante) > 0 else 0,
            stats_mandante['Empates como Mandante'] if len(jogos_mandante) > 0 else 0,
            stats_mandante['Derrotas como Mandante'] if len(jogos_mandante) > 0 else 0
        ]
        visitante_dados = [
            stats_visitante['Vitórias como Visitante'] if len(jogos_visitante) > 0 else 0,
            stats_visitante['Empates como Visitante'] if len(jogos_visitante) > 0 else 0,
            stats_visitante['Derrotas como Visitante'] if len(jogos_visitante) > 0 else 0
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categorias))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mandante_dados, width, label=f'{time_mandante_filtro} (Mandante)', color='#2563eb', alpha=0.8)
        bars2 = ax.bar(x + width/2, visitante_dados, width, label=f'{time_visitante_filtro} (Visitante)', color='#dc2626', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Resultados')
        ax.set_ylabel('Quantidade de Jogos')
        ax.set_title('Comparação de Desempenho entre os Times Selecionados')
        ax.set_xticks(x)
        ax.set_xticklabels(categorias)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)

elif opcao == "Análise de Placar":
    st.subheader(f"Análise de Placar: {time_mandante_filtro} vs {time_visitante_filtro}")
    
    jogos_mandante = raw_df[raw_df['mandante'] == time_mandante_filtro]
    jogos_visitante = raw_df[raw_df['visitante'] == time_visitante_filtro]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"⚽ {time_mandante_filtro} como Mandante")
        if len(jogos_mandante) > 0:
            gols_feitos_mandante = jogos_mandante["mandante_Placar"].sum()
            gols_sofridos_mandante = jogos_mandante["visitante_Placar"].sum()
            media_gols_feitos = jogos_mandante["mandante_Placar"].mean()
            media_gols_sofridos = jogos_mandante["visitante_Placar"].mean()
            
            st.metric("🥅 Total de Gols Feitos", int(gols_feitos_mandante))
            st.metric("🛡️ Total de Gols Sofridos", int(gols_sofridos_mandante))
            st.metric("📊 Média de Gols Feitos", f"{media_gols_feitos:.2f}")
            st.metric("📉 Média de Gols Sofridos", f"{media_gols_sofridos:.2f}")
        
            saldo = gols_feitos_mandante - gols_sofridos_mandante
            st.metric("⚖️ Saldo de Gols", int(saldo), delta=None if saldo == 0 else int(saldo))
        else:
            st.warning("Nenhum jogo como mandante encontrado")
    
    with col2:
        st.subheader(f"✈️ {time_visitante_filtro} como Visitante")
        if len(jogos_visitante) > 0:
            gols_feitos_visitante = jogos_visitante["visitante_Placar"].sum()
            gols_sofridos_visitante = jogos_visitante["mandante_Placar"].sum()
            media_gols_feitos = jogos_visitante["visitante_Placar"].mean()
            media_gols_sofridos = jogos_visitante["mandante_Placar"].mean()
            
            st.metric("🥅 Total de Gols Feitos", int(gols_feitos_visitante))
            st.metric("🛡️ Total de Gols Sofridos", int(gols_sofridos_visitante))
            st.metric("📊 Média de Gols Feitos", f"{media_gols_feitos:.2f}")
            st.metric("📉 Média de Gols Sofridos", f"{media_gols_sofridos:.2f}")
            
            saldo = gols_feitos_visitante - gols_sofridos_visitante
            st.metric("⚖️ Saldo de Gols", int(saldo), delta=None if saldo == 0 else int(saldo))
        else:
            st.warning("Nenhum jogo como visitante encontrado")

    if len(jogos_mandante) > 0 or len(jogos_visitante) > 0:
        st.subheader("📊 Comparação de Gols")
        
        gols_feitos = [
            jogos_mandante["mandante_Placar"].sum() if len(jogos_mandante) > 0 else 0,
            jogos_visitante["visitante_Placar"].sum() if len(jogos_visitante) > 0 else 0
        ]
        gols_sofridos = [
            jogos_mandante["visitante_Placar"].sum() if len(jogos_mandante) > 0 else 0,
            jogos_visitante["mandante_Placar"].sum() if len(jogos_visitante) > 0 else 0
        ]
        
        times = [f"{time_mandante_filtro}\n(Mandante)", f"{time_visitante_filtro}\n(Visitante)"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(times))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gols_feitos, width, label='Gols Feitos', color='#22c55e', alpha=0.8)
        bars2 = ax.bar(x + width/2, gols_sofridos, width, label='Gols Sofridos', color='#ef4444', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Número de Gols')
        ax.set_title('Comparação de Gols: Feitos vs Sofridos')
        ax.set_xticks(x)
        ax.set_xticklabels(times)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
    
        st.subheader("📈 Distribuição de Placares")
        
        tab1, tab2 = st.tabs([f"{time_mandante_filtro} (Mandante)", f"{time_visitante_filtro} (Visitante)"])
        
        with tab1:
            if len(jogos_mandante) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.hist(jogos_mandante["mandante_Placar"], bins=range(0, jogos_mandante["mandante_Placar"].max()+2), 
                        alpha=0.7, color='#22c55e', edgecolor='black')
                ax1.set_title('Distribuição de Gols Feitos')
                ax1.set_xlabel('Gols')
                ax1.set_ylabel('Frequência')
                
                ax2.hist(jogos_mandante["visitante_Placar"], bins=range(0, jogos_mandante["visitante_Placar"].max()+2), 
                        alpha=0.7, color='#ef4444', edgecolor='black')
                ax2.set_title('Distribuição de Gols Sofridos')
                ax2.set_xlabel('Gols')
                ax2.set_ylabel('Frequência')
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Sem dados para mostrar distribuição")
        
        with tab2:
            if len(jogos_visitante) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.hist(jogos_visitante["visitante_Placar"], bins=range(0, jogos_visitante["visitante_Placar"].max()+2), 
                        alpha=0.7, color='#22c55e', edgecolor='black')
                ax1.set_title('Distribuição de Gols Feitos')
                ax1.set_xlabel('Gols')
                ax1.set_ylabel('Frequência')
                
                ax2.hist(jogos_visitante["mandante_Placar"], bins=range(0, jogos_visitante["mandante_Placar"].max()+2), 
                        alpha=0.7, color='#ef4444', edgecolor='black')
                ax2.set_title('Distribuição de Gols Sofridos')
                ax2.set_xlabel('Gols')
                ax2.set_ylabel('Frequência')
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Sem dados para mostrar distribuição")

elif opcao == "Distribuição de Placar":
    st.subheader(f"Distribuição de Placar: {time_mandante_filtro} vs {time_visitante_filtro}")
    
    jogos_mandante = raw_df[raw_df['mandante'] == time_mandante_filtro]
    jogos_visitante = raw_df[raw_df['visitante'] == time_visitante_filtro]
    
    if len(jogos_mandante) == 0 and len(jogos_visitante) == 0:
        st.warning("Nenhum jogo encontrado para os times selecionados.")
    else:
        tab1, tab2, tab3 = st.tabs([
            f"{time_mandante_filtro} (Mandante)", 
            f"{time_visitante_filtro} (Visitante)",
            "Comparação"
        ])
        
        with tab1:
            if len(jogos_mandante) > 0:
                st.subheader(f"📊 {time_mandante_filtro} jogando como Mandante")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    data_to_plot = [jogos_mandante["mandante_Placar"], jogos_mandante["visitante_Placar"]]
                    box_plot = ax.boxplot(data_to_plot, labels=["Gols Feitos", "Gols Sofridos"], patch_artist=True)
                    
                    colors = ['#22c55e', '#ef4444']
                    for patch, color in zip(box_plot['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_title(f'Distribuição de Gols - {time_mandante_filtro}')
                    ax.set_ylabel('Número de Gols')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    st.write("📈 **Estatísticas dos Gols Feitos:**")
                    gols_feitos_stats = jogos_mandante["mandante_Placar"].describe()
                    st.write(f"• Média: {gols_feitos_stats['mean']:.2f}")
                    st.write(f"• Mediana: {gols_feitos_stats['50%']:.0f}")
                    st.write(f"• Mínimo: {gols_feitos_stats['min']:.0f}")
                    st.write(f"• Máximo: {gols_feitos_stats['max']:.0f}")
                    st.write(f"• Desvio Padrão: {gols_feitos_stats['std']:.2f}")
                    
                    st.write("📉 **Estatísticas dos Gols Sofridos:**")
                    gols_sofridos_stats = jogos_mandante["visitante_Placar"].describe()
                    st.write(f"• Média: {gols_sofridos_stats['mean']:.2f}")
                    st.write(f"• Mediana: {gols_sofridos_stats['50%']:.0f}")
                    st.write(f"• Mínimo: {gols_sofridos_stats['min']:.0f}")
                    st.write(f"• Máximo: {gols_sofridos_stats['max']:.0f}")
                    st.write(f"• Desvio Padrão: {gols_sofridos_stats['std']:.2f}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                ax1.hist(jogos_mandante["mandante_Placar"], bins=range(0, int(jogos_mandante["mandante_Placar"].max())+2), 
                        alpha=0.7, color='#22c55e', edgecolor='black', density=True)
                ax1.set_title('Distribuição de Gols Feitos (Densidade)')
                ax1.set_xlabel('Gols Feitos')
                ax1.set_ylabel('Densidade')
                ax1.grid(True, alpha=0.3)
                
                ax2.hist(jogos_mandante["visitante_Placar"], bins=range(0, int(jogos_mandante["visitante_Placar"].max())+2), 
                        alpha=0.7, color='#ef4444', edgecolor='black', density=True)
                ax2.set_title('Distribuição de Gols Sofridos (Densidade)')
                ax2.set_xlabel('Gols Sofridos')
                ax2.set_ylabel('Densidade')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(f"Nenhum jogo de {time_mandante_filtro} como mandante encontrado.")
        
        with tab2:
            if len(jogos_visitante) > 0:
                st.subheader(f"📊 {time_visitante_filtro} jogando como Visitante")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    data_to_plot = [jogos_visitante["visitante_Placar"], jogos_visitante["mandante_Placar"]]
                    box_plot = ax.boxplot(data_to_plot, labels=["Gols Feitos", "Gols Sofridos"], patch_artist=True)
                    
                    colors = ['#22c55e', '#ef4444']
                    for patch, color in zip(box_plot['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_title(f'Distribuição de Gols - {time_visitante_filtro}')
                    ax.set_ylabel('Número de Gols')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    st.write("📈 **Estatísticas dos Gols Feitos:**")
                    gols_feitos_stats = jogos_visitante["visitante_Placar"].describe()
                    st.write(f"• Média: {gols_feitos_stats['mean']:.2f}")
                    st.write(f"• Mediana: {gols_feitos_stats['50%']:.0f}")
                    st.write(f"• Mínimo: {gols_feitos_stats['min']:.0f}")
                    st.write(f"• Máximo: {gols_feitos_stats['max']:.0f}")
                    st.write(f"• Desvio Padrão: {gols_feitos_stats['std']:.2f}")
                    
                    st.write("📉 **Estatísticas dos Gols Sofridos:**")
                    gols_sofridos_stats = jogos_visitante["mandante_Placar"].describe()
                    st.write(f"• Média: {gols_sofridos_stats['mean']:.2f}")
                    st.write(f"• Mediana: {gols_sofridos_stats['50%']:.0f}")
                    st.write(f"• Mínimo: {gols_sofridos_stats['min']:.0f}")
                    st.write(f"• Máximo: {gols_sofridos_stats['max']:.0f}")
                    st.write(f"• Desvio Padrão: {gols_sofridos_stats['std']:.2f}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
                ax1.hist(jogos_visitante["visitante_Placar"], bins=range(0, int(jogos_visitante["visitante_Placar"].max())+2), 
                        alpha=0.7, color='#22c55e', edgecolor='black', density=True)
                ax1.set_title('Distribuição de Gols Feitos (Densidade)')
                ax1.set_xlabel('Gols Feitos')
                ax1.set_ylabel('Densidade')
                ax1.grid(True, alpha=0.3)
                
                ax2.hist(jogos_visitante["mandante_Placar"], bins=range(0, int(jogos_visitante["mandante_Placar"].max())+2), 
                        alpha=0.7, color='#ef4444', edgecolor='black', density=True)
                ax2.set_title('Distribuição de Gols Sofridos (Densidade)')
                ax2.set_xlabel('Gols Sofridos')
                ax2.set_ylabel('Densidade')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(f"Nenhum jogo de {time_visitante_filtro} como visitante encontrado.")
        
        with tab3:
            if len(jogos_mandante) > 0 and len(jogos_visitante) > 0:
                st.subheader("⚖️ Comparação Direta")
            
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    media_gols_mandante = jogos_mandante["mandante_Placar"].mean()
                    media_gols_visitante = jogos_visitante["visitante_Placar"].mean()
                    st.metric(
                        "🥅 Média de Gols Feitos",
                        f"{time_mandante_filtro}: {media_gols_mandante:.2f}",
                        delta=float(f"{media_gols_mandante - media_gols_visitante:.2f}") if media_gols_mandante != media_gols_visitante else None
                    )
                    st.write(f"{time_visitante_filtro}: {media_gols_visitante:.2f}")
                
                with col2:
                    media_sofridos_mandante = jogos_mandante["visitante_Placar"].mean()
                    media_sofridos_visitante = jogos_visitante["mandante_Placar"].mean()
                    st.metric(
                        "🛡️ Média de Gols Sofridos",
                        f"{time_mandante_filtro}: {media_sofridos_mandante:.2f}",
                        delta=float(f"{media_sofridos_mandante - media_sofridos_visitante:.2f}") if media_sofridos_mandante != media_sofridos_visitante else None
                    )
                    st.write(f"{time_visitante_filtro}: {media_sofridos_visitante:.2f}")
                
                with col3:
                    saldo_mandante = media_gols_mandante - media_sofridos_mandante
                    saldo_visitante = media_gols_visitante - media_sofridos_visitante
                    st.metric(
                        "⚖️ Saldo Médio",
                        f"{time_mandante_filtro}: {saldo_mandante:.2f}",
                        delta=float(f"{saldo_mandante - saldo_visitante:.2f}") if saldo_mandante != saldo_visitante else None
                    )
                    st.write(f"{time_visitante_filtro}: {saldo_visitante:.2f}")
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                ax1.hist(jogos_mandante["mandante_Placar"], alpha=0.7, label=f'{time_mandante_filtro} (M)', color='#2563eb', density=True)
                ax1.hist(jogos_visitante["visitante_Placar"], alpha=0.7, label=f'{time_visitante_filtro} (V)', color='#dc2626', density=True)
                ax1.set_title('Comparação: Gols Feitos')
                ax1.set_xlabel('Gols')
                ax1.set_ylabel('Densidade')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
                ax2.hist(jogos_mandante["visitante_Placar"], alpha=0.7, label=f'{time_mandante_filtro} (M)', color='#2563eb', density=True)
                ax2.hist(jogos_visitante["mandante_Placar"], alpha=0.7, label=f'{time_visitante_filtro} (V)', color='#dc2626', density=True)
                ax2.set_title('Comparação: Gols Sofridos')
                ax2.set_xlabel('Gols')
                ax2.set_ylabel('Densidade')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                data_gols_feitos = [jogos_mandante["mandante_Placar"], jogos_visitante["visitante_Placar"]]
                box1 = ax3.boxplot(data_gols_feitos, labels=[f'{time_mandante_filtro}\n(M)', f'{time_visitante_filtro}\n(V)'], patch_artist=True)
                colors = ['#2563eb', '#dc2626']
                for patch, color in zip(box1['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax3.set_title('Box Plot: Gols Feitos')
                ax3.set_ylabel('Gols')
                ax3.grid(True, alpha=0.3)
                
                data_gols_sofridos = [jogos_mandante["visitante_Placar"], jogos_visitante["mandante_Placar"]]
                box2 = ax4.boxplot(data_gols_sofridos, labels=[f'{time_mandante_filtro}\n(M)', f'{time_visitante_filtro}\n(V)'], patch_artist=True)
                for patch, color in zip(box2['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax4.set_title('Box Plot: Gols Sofridos')
                ax4.set_ylabel('Gols')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Dados insuficientes para comparação. Ambos os times precisam ter jogos registrados.")

elif opcao == "Método do Cotovelo (K-Means)":
    st.subheader("Método do Cotovelo para K-Means")

    usar_filtro = st.checkbox(
        "🎯 Analisar apenas os times selecionados", 
        value=False,
        help="Marque para analisar apenas os times selecionados, ou desmarque para analisar todos os times"
    )
    
    if usar_filtro:
        times_para_analise = [time_mandante_filtro, time_visitante_filtro]
        dados_filtrados = raw_df[raw_df["mandante"].isin(times_para_analise)]
        
        if len(dados_filtrados) == 0:
            st.warning("Nenhum dado encontrado para os times selecionados.")
            st.stop()
        
        st.info(f"📊 Analisando: {time_mandante_filtro} e {time_visitante_filtro}")
        dados_cluster = dados_filtrados.groupby("mandante").agg({
            "mandante_Placar": "mean",
            "visitante_Placar": "mean", 
            "saldo_gols": "mean",
            "vencedor": lambda x: (x == "Mandante").mean()
        }).reset_index()
    else:
        st.info("📊 Analisando todos os times do campeonato")
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
    
    if len(X) < 2:
        st.warning("Dados insuficientes para análise de clustering (mínimo 2 times necessários).")
        st.stop()
    
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X)


    max_clusters = min(10, len(X))
    wcss = []
    
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, max_clusters + 1), wcss, marker='o', linewidth=2, markersize=8)
    ax.set_title(f'Método do Cotovelo ({len(dados_cluster)} times analisados)')
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
    ax.grid(True, alpha=0.3)
    
    if len(wcss) >= 3:
        derivatives = []
        for i in range(1, len(wcss)-1):
            derivatives.append(wcss[i-1] - 2*wcss[i] + wcss[i+1])
        
        if derivatives:
            cotovelo_idx = derivatives.index(max(derivatives)) + 2 
            ax.axvline(x=cotovelo_idx, color='red', linestyle='--', alpha=0.7, 
                      label=f'Cotovelo sugerido: K={cotovelo_idx}')
            ax.legend()
    for i, wcss_val in enumerate(wcss):
        ax.annotate(f'{wcss_val:.2f}', (i+1, wcss_val), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    wcss_df = pd.DataFrame({
        'K (Clusters)': range(1, max_clusters + 1),
        'WCSS': wcss,
        'Redução WCSS': [0] + [wcss[i-1] - wcss[i] for i in range(1, len(wcss))]
    })
    
    st.subheader("📊 Tabela de Valores WCSS")
    st.dataframe(wcss_df, use_container_width=True, hide_index=True)
    
    if len(wcss) >= 3:
        st.subheader("💡 Análise do Cotovelo")
        st.write(f"""
        **Interpretação:**
        - O "cotovelo" indica o ponto onde adicionar mais clusters não reduz significativamente o WCSS
        - Clusters ideais sugeridos: **K = {cotovelo_idx if 'cotovelo_idx' in locals() else 'N/A'}**
        - Para {len(dados_cluster)} times, recomenda-se entre 2-4 clusters
        """)
        
        if usar_filtro:
            st.info(f"💡 Com apenas 2 times selecionados ({time_mandante_filtro} e {time_visitante_filtro}), o clustering terá utilidade limitada. Considere desmarcar o filtro para uma análise mais abrangente.")
    else:
        st.warning("Dados insuficientes para determinar o número ideal de clusters.")

elif opcao == "Clusterização dos Times":
    st.subheader("Clusterização dos Times")
    
    usar_filtro = st.checkbox(
        "🎯 Analisar apenas os times selecionados", 
        value=False,
        help="Marque para analisar apenas os times selecionados, ou desmarque para analisar todos os times",
        key="cluster_filter"
    )
    
    if usar_filtro:

        times_para_analise = [time_mandante_filtro, time_visitante_filtro]
        dados_filtrados = raw_df[raw_df["mandante"].isin(times_para_analise)]
        
        if len(dados_filtrados) == 0:
            st.warning("Nenhum dado encontrado para os times selecionados.")
            st.stop()
        
        st.info(f"📊 Analisando: {time_mandante_filtro} e {time_visitante_filtro}")
        dados_cluster = dados_filtrados.groupby("mandante").agg({
            "mandante_Placar": "mean",
            "visitante_Placar": "mean",
            "saldo_gols": "mean",
            "vencedor": lambda x: (x == "Mandante").mean()
        }).reset_index()
    else:
        st.info("📊 Analisando todos os times do campeonato")
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
    
    if len(X) < 2:
        st.warning("Dados insuficientes para análise de clustering (mínimo 2 times necessários).")
        st.stop()
    
    n_clusters = min(3, len(X))
    if len(X) == 2:
        n_clusters = 2
        st.info("⚠️ Com apenas 2 times, o clustering criará 2 clusters (um para cada time)")
    
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    dados_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    cores = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6"][:n_clusters]
    for i in range(n_clusters):
        cluster = dados_cluster[dados_cluster["Cluster"] == i]
        ax1.scatter(cluster["media_gols_feitos"], cluster["media_saldo_gols"], 
                   color=cores[i], label=f"Cluster {i}", s=100, alpha=0.7, edgecolors='black')
        
        for idx, row in cluster.iterrows():
            ax1.annotate(row["mandante"], 
                        (row["media_gols_feitos"], row["media_saldo_gols"]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, ha='left')
    
    ax1.set_xlabel("Média de Gols Feitos")
    ax1.set_ylabel("Média de Saldo de Gols")
    ax1.set_title("Clusterização: Gols Feitos vs Saldo de Gols")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i in range(n_clusters):
        cluster = dados_cluster[dados_cluster["Cluster"] == i]
        ax2.scatter(cluster["taxa_vitorias"], cluster["media_gols_sofridos"], 
                   color=cores[i], label=f"Cluster {i}", s=100, alpha=0.7, edgecolors='black')
        
        for idx, row in cluster.iterrows():
            ax2.annotate(row["mandante"], 
                        (row["taxa_vitorias"], row["media_gols_sofridos"]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, ha='left')
    
    ax2.set_xlabel("Taxa de Vitórias")
    ax2.set_ylabel("Média de Gols Sofridos") 
    ax2.set_title("Clusterização: Taxa de Vitórias vs Gols Sofridos")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🏷️ Análise dos Clusters Identificados")
    
    for i in range(n_clusters):
        cluster_data = dados_cluster[dados_cluster["Cluster"] == i]
        times_no_cluster = cluster_data["mandante"].tolist()
        
        media_gols_feitos = cluster_data["media_gols_feitos"].mean()
        media_gols_sofridos = cluster_data["media_gols_sofridos"].mean()
        media_saldo = cluster_data["media_saldo_gols"].mean()
        media_vitorias = cluster_data["taxa_vitorias"].mean()
        
        if media_saldo > 0.5 and media_vitorias > 0.4:
            perfil = "🔴 **Alto Desempenho**"
            descricao = "Times com excelente saldo de gols e alta taxa de vitórias"
        elif media_saldo > 0 and media_vitorias > 0.25:
            perfil = "🔵 **Desempenho Intermediário**"
            descricao = "Times com desempenho médio, saldo positivo moderado"
        else:
            perfil = "🟢 **Baixo Desempenho**"
            descricao = "Times com dificuldades, saldo baixo e poucas vitórias"
        
        with st.expander(f"Cluster {i} - {perfil} ({len(times_no_cluster)} time{'s' if len(times_no_cluster) != 1 else ''})"):
            st.write(f"**{descricao}**")
            st.write(f"**Times:** {', '.join(times_no_cluster)}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⚽ Gols Feitos", f"{media_gols_feitos:.2f}")
            with col2:
                st.metric(" ️ Gols Sofridos", f"{media_gols_sofridos:.2f}")
            with col3:
                st.metric("⚖️ Saldo Médio", f"{media_saldo:.2f}")
            with col4:
                st.metric("🏆 Taxa Vitórias", f"{media_vitorias:.1%}")

    st.subheader("📊 Dados Detalhados por Time")
    
    tabela_display = dados_cluster.copy()
    tabela_display["taxa_vitorias"] = tabela_display["taxa_vitorias"].apply(lambda x: f"{x:.1%}")
    tabela_display = tabela_display.round(2)
    
    tabela_display = tabela_display.rename(columns={
        "mandante": "Time",
        "media_gols_feitos": "Gols Feitos (Média)",
        "media_gols_sofridos": "Gols Sofridos (Média)",
        "media_saldo_gols": "Saldo de Gols (Média)",
        "taxa_vitorias": "Taxa de Vitórias",
        "Cluster": "Cluster"
    })
    
    if usar_filtro:
        st.dataframe(tabela_display, use_container_width=True, hide_index=True)
    else:
        def highlight_selected_teams(row):
            if row["Time"] in [time_mandante_filtro, time_visitante_filtro]:
                return ['background-color: #fef3c7'] * len(row)
            return [''] * len(row)
        
        styled_df = tabela_display.style.apply(highlight_selected_teams, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.info(f"💡 Os times selecionados ({time_mandante_filtro} e {time_visitante_filtro}) estão destacados em amarelo na tabela.")
    
    if not usar_filtro:
        st.subheader("🎯 Análise dos Times Selecionados")
        
        time_mandante_cluster = dados_cluster[dados_cluster["mandante"] == time_mandante_filtro]["Cluster"].iloc[0] if len(dados_cluster[dados_cluster["mandante"] == time_mandante_filtro]) > 0 else None
        time_visitante_cluster = dados_cluster[dados_cluster["mandante"] == time_visitante_filtro]["Cluster"].iloc[0] if len(dados_cluster[dados_cluster["mandante"] == time_visitante_filtro]) > 0 else None
        
        col1, col2 = st.columns(2)
        
        with col1:
            if time_mandante_cluster is not None:
                st.write(f"🏠 **{time_mandante_filtro}** pertence ao **Cluster {time_mandante_cluster}**")
            else:
                st.write(f"⚠️ {time_mandante_filtro} não possui dados suficientes como mandante")
        
        with col2:
            if time_visitante_cluster is not None:
                st.write(f"✈️ **{time_visitante_filtro}** pertence ao **Cluster {time_visitante_cluster}**")
            else:
                st.write(f"⚠️ {time_visitante_filtro} não possui dados suficientes como mandante")
        
        if time_mandante_cluster is not None and time_visitante_cluster is not None:
            if time_mandante_cluster == time_visitante_cluster:
                st.success("🤝 Ambos os times estão no mesmo cluster, indicando perfis de desempenho similares!")
            else:
                st.info("⚔️ Os times estão em clusters diferentes, indicando perfis de desempenho distintos.")
    
    if usar_filtro and len(dados_cluster) == 2:
        st.info("💡 Para uma análise mais completa, desmarque a opção de filtro para ver como estes times se comparam com todos os outros times do campeonato.")

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
                        st.metric("🎮 Total de Jogos", int(total_jogos))
                    with col2:
                        st.metric(f"🏆 Vitórias {time_mandante}", int(vitorias_mandante))
                    with col3:
                        st.metric("🤝 Empates", int(empates))
                    with col4:
                        st.metric(f"🏆 Vitórias {time_visitante}", int(vitorias_visitante))
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