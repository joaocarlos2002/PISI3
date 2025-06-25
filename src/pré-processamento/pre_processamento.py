import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import os

def carregar_dados(path_csv):
    df = pd.read_csv(path_csv)
    df.columns = df.columns.str.strip()
    return df

def criar_variaveis_auxiliares(df):
    df['saldo_gols'] = df['mandante_Placar'] - df['visitante_Placar']
    df['vencedor'] = df.apply(
        lambda row: 'Mandante' if row['mandante_Placar'] > row['visitante_Placar']
        else ('Visitante' if row['mandante_Placar'] < row['visitante_Placar'] else 'Empate'),
        axis=1
    )
    return df

def rolling_form_features(df, n_games=10):
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce', dayfirst=True)
        main_col = 'data'
    elif 'rodada' in df.columns:
        main_col = 'rodada'
    elif 'rodata' in df.columns:
        main_col = 'rodata'
    else:
        raise ValueError('Nenhuma coluna de data/rodada encontrada.')
    df = df.sort_values(main_col).reset_index(drop=True)
    df_form = df.copy()
    df_form['mandante_points'] = df_form.apply(lambda r: 3 if r['vencedor'] == 'Mandante' else (1 if r['vencedor'] == 'Empate' else 0), axis=1)
    df_form['visitante_points'] = df_form.apply(lambda r: 3 if r['vencedor'] == 'Visitante' else (1 if r['vencedor'] == 'Empate' else 0), axis=1)
    team_stats_list = []
    for idx, row in df_form.iterrows():
        team_stats_list.append({
            'team': row['mandante'],
            main_col: row[main_col],
            'goals_scored': row['mandante_Placar'],
            'goals_conceded': row['visitante_Placar'],
            'points': row['mandante_points']
        })
        team_stats_list.append({
            'team': row['visitante'],
            main_col: row[main_col],
            'goals_scored': row['visitante_Placar'],
            'goals_conceded': row['mandante_Placar'],
            'points': row['visitante_points']
        })
    df_team_performances = pd.DataFrame(team_stats_list)
    df_team_performances = df_team_performances.sort_values(by=['team', main_col]).reset_index(drop=True)
    df_team_performances[f'rolling_goals_scored_{n_games}'] = df_team_performances.groupby('team')['goals_scored'].transform(lambda x: x.rolling(window=n_games, min_periods=1).mean().shift(1))
    df_team_performances[f'rolling_goals_conceded_{n_games}'] = df_team_performances.groupby('team')['goals_conceded'].transform(lambda x: x.rolling(window=n_games, min_periods=1).mean().shift(1))
    df_team_performances[f'rolling_points_{n_games}'] = df_team_performances.groupby('team')['points'].transform(lambda x: x.rolling(window=n_games, min_periods=1).mean().shift(1))
    df_team_performances.fillna(0, inplace=True)
    df_ml = df.merge(
        df_team_performances[['team', main_col, f'rolling_goals_scored_{n_games}', f'rolling_goals_conceded_{n_games}', f'rolling_points_{n_games}']],
        left_on=['mandante', main_col],
        right_on=['team', main_col],
        how='left',
        suffixes=('', '_mandante_form')
    )
    df_ml.drop(columns=['team'], inplace=True)
    df_ml = df_ml.merge(
        df_team_performances[['team', main_col, f'rolling_goals_scored_{n_games}', f'rolling_goals_conceded_{n_games}', f'rolling_points_{n_games}']],
        left_on=['visitante', main_col],
        right_on=['team', main_col],
        how='left',
        suffixes=('', '_visitante_form')
    )
    df_ml.drop(columns=['team'], inplace=True)
    df_ml[f'diff_rolling_goals_scored_{n_games}'] = df_ml[f'rolling_goals_scored_{n_games}'] - df_ml[f'rolling_goals_scored_{n_games}_visitante_form']
    df_ml[f'diff_rolling_goals_conceded_{n_games}'] = df_ml[f'rolling_goals_conceded_{n_games}'] - df_ml[f'rolling_goals_conceded_{n_games}_visitante_form']
    df_ml[f'diff_rolling_points_{n_games}'] = df_ml[f'rolling_points_{n_games}'] - df_ml[f'rolling_points_{n_games}_visitante_form']
    return df_ml

def pipeline_preprocessamento(path_csv, path_pkl):
    df = carregar_dados(path_csv)
    df = criar_variaveis_auxiliares(df)
    df_ml = rolling_form_features(df, n_games=10)
    features = [
        'rolling_goals_scored_10', 'rolling_goals_conceded_10', 'rolling_points_10',
        'rolling_goals_scored_10_visitante_form', 'rolling_goals_conceded_10_visitante_form', 'rolling_points_10_visitante_form',
        'diff_rolling_goals_scored_10', 'diff_rolling_goals_conceded_10', 'diff_rolling_points_10'
    ]
    df_ml = df_ml.dropna(subset=features + ['vencedor'])
    X = df_ml[features]
    le = LabelEncoder()
    y = le.fit_transform(df_ml['vencedor'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    dados_consolidados = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_res,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoder': le,
        'features': features
    }
    os.makedirs(os.path.dirname(path_pkl), exist_ok=True)
    with open(path_pkl, 'wb') as f:
        pickle.dump(dados_consolidados, f)
    print(f'Pré-processamento concluído. Total de features: {len(features)}')

if __name__ == '__main__':
    pipeline_preprocessamento(
        'src/data/campeonato-brasileiro.csv',
        'src/data/data-aprendizado/dados_consolidados.pkl'
    )


