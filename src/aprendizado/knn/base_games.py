import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def progress_bar(current, total, desc="Processando"):
    bar_length = 50
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percent = progress * 100
    print(f'\r{desc}: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()

class KNNGamePredictor:
    def __init__(self, data_path='src/data/data-aprendizado/dados_consolidados.pkl'):
        self.data_path = data_path
        self.model = None
        self.best_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}
        self.figures_path = 'src/data/figuras/knn'
        self.class_names = ['Empate', 'Mandante', 'Visitante']

    def load_data(self):
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            self.X_train = data['X_train']
            self.X_test = data['X_test']
            self.y_train = data['y_train']
            self.y_test = data['y_test']
            self.feature_names = data.get('features', None)
            return True
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False

    def optimize_hyperparameters(self, cv_folds=5, scoring='accuracy'):
        print("🔧 Otimizando hiperparâmetros do KNN...")
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski'],
            'p': [1, 2]
        }

        combinations = [
            {'n_neighbors': k, 'weights': w, 'metric': m, 'p': p}
            for k in param_grid['n_neighbors']
            for w in param_grid['weights']
            for m in param_grid['metric']
            for p in param_grid['p']
        ]

        total_combinations = len(combinations)
        for i in range(total_combinations + 1):
            progress_bar(i, total_combinations, "Testando combinações")
            time.sleep(0.01)

        grid = GridSearchCV(KNeighborsClassifier(), combinations, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42), scoring=scoring, n_jobs=-1, verbose=0)
        grid.fit(self.X_train, self.y_train)
        self.best_params = grid.best_params_
        self.model = grid.best_estimator_
        print(f"✅ Melhor acurácia CV: {grid.best_score_:.4f}")
        return grid.best_score_

    def train_model(self, **kwargs):
        if not self.model:
            self.model = KNeighborsClassifier(**(kwargs or {'n_neighbors': 5}))
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        self.results = {
            'train_accuracy': accuracy_score(self.y_train, self.model.predict(self.X_train)),
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
            'best_params': self.best_params
        }
        print(classification_report(self.y_test, y_pred, target_names=['Empate', 'Mandante', 'Visitante']))
        return self.results

    def create_figures_directory(self):
        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path)

    def plot_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusão - KNN')
        plt.xlabel('Predição')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_validation_curve(self):
        k_range = range(1, 21)
        weights = self.best_params.get('weights', 'uniform') if self.best_params else 'uniform'
        metric = self.best_params.get('metric', 'minkowski') if self.best_params else 'minkowski'
        p = self.best_params.get('p', 2) if self.best_params else 2
        best_k = self.best_params.get('n_neighbors', 5) if self.best_params else 5
        train_scores, val_scores = validation_curve(
            KNeighborsClassifier(weights=weights, metric=metric, p=p),
            self.X_train, self.y_train, 
            param_name='n_neighbors', param_range=k_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, train_mean, 'o-', color='blue', label='Treino')
        plt.fill_between(k_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(k_range, val_mean, 'o-', color='red', label='Validação')
        plt.fill_between(k_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        plt.axvline(x=best_k, color='green', linestyle='--', label=f'Melhor K = {best_k}')
        plt.xlabel('Número de Vizinhos (K)')
        plt.ylabel('Acurácia')
        plt.title('Curva de Validação - KNN')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/validation_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metrics_comparison(self):
        metrics = ['train_accuracy', 'test_accuracy', 'precision', 'recall', 'f1_score']
        values = [self.results[metric] for metric in metrics if metric in self.results]
        metric_labels = ['Acurácia Treino', 'Acurácia Teste', 'Precisão', 'Recall', 'F1-Score']
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_labels, values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum'])
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.ylim(0, 1.1)
        plt.ylabel('Score')
        plt.title('Métricas de Performance - KNN')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_hyperparameter_importance(self):
        if not self.best_params:
            default_params = {
                'n_neighbors': 5,
                'weights': 'uniform',
                'metric': 'minkowski',
                'p': 2
            }
            param_names = list(default_params.keys())
            param_values = [str(v) for v in default_params.values()]
            title_suffix = " (Parâmetros Padrão)"
        else:
            param_names = list(self.best_params.keys())
            param_values = [str(v) for v in self.best_params.values()]
            title_suffix = ""
        plt.figure(figsize=(10, 6))
        bars = plt.barh(param_names, range(len(param_names)), color='lightblue')
        for i, (bar, value) in enumerate(zip(bars, param_values)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    value, ha='left', va='center', fontweight='bold')
        plt.xlabel('Ordem de Importância')
        plt.title(f'Melhores Hiperparâmetros - KNN{title_suffix}')
        plt.xlim(0, len(param_names) + 1)
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/best_hyperparameters.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_class_distribution(self):
        train_counts = np.bincount(self.y_train)
        test_counts = np.bincount(self.y_test)
        x = np.arange(len(self.class_names))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, train_counts, width, label='Treino', color='lightblue')
        plt.bar(x + width/2, test_counts, width, label='Teste', color='lightcoral')
        for i in range(len(self.class_names)):
            plt.text(i - width/2, train_counts[i] + 10, str(train_counts[i]), 
                    ha='center', va='bottom', fontweight='bold')
            plt.text(i + width/2, test_counts[i] + 10, str(test_counts[i]), 
                    ha='center', va='bottom', fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Número de Amostras')
        plt.title('Distribuição das Classes nos Conjuntos de Treino e Teste')
        plt.xticks(x, self.class_names)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_kmeans_clusters(self, X, y, feature_names, save_path, n_clusters=3):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7, s=60)
        centers = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroides')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('Clusterização K-Means (PCA 2D)')
        plt.legend(*scatter.legend_elements(), title="Cluster")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cluster_distribution(self, X, save_path, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)
        unique, counts = np.unique(clusters, return_counts=True)
        plt.figure(figsize=(7, 5))
        plt.bar([f'Cluster {i}' for i in unique], counts, color=['#2563eb', '#facc15', '#22c55e'])
        plt.xlabel('Cluster')
        plt.ylabel('Nº de Amostras')
        plt.title('Distribuição dos Clusters (K-Means)')
        for i, v in enumerate(counts):
            plt.text(i, v + 2, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_feature_importance(self):
        print("\n🔍 ANÁLISE DE IMPORTÂNCIA DAS FEATURES (KNN):")
        print("=" * 60)
        
        if self.feature_names is None:
            print("❌ Nomes das features não disponíveis")
            return None
        
        base_accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        feature_importance = {}
        
        total_features = len(self.feature_names)
        print(f"📊 Analisando {total_features} features...")
        
        for i, feature_name in enumerate(self.feature_names):
            progress_bar(i + 1, total_features, "Calculando importância")
            time.sleep(0.02)
            
            X_train_reduced = np.delete(self.X_train, i, axis=1)
            X_test_reduced = np.delete(self.X_test, i, axis=1)
            
            temp_model = KNeighborsClassifier(**self.best_params if self.best_params else {'n_neighbors': 5})
            temp_model.fit(X_train_reduced, self.y_train)
            
            reduced_accuracy = accuracy_score(self.y_test, temp_model.predict(X_test_reduced))
            importance = base_accuracy - reduced_accuracy
            feature_importance[feature_name] = importance
        
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"📊 Acurácia base (todas as features): {base_accuracy:.4f}")
        print("\n🎯 RANKING DE IMPORTÂNCIA DAS FEATURES:")
        print("-" * 60)
        
        for i, (feature, importance) in enumerate(sorted_importance, 1):
            impact = "🔴" if importance > 0.01 else "🟡" if importance > 0.005 else "🟢"
            print(f"{i:2d}. {impact} {feature:<35} | Importância: {importance:+.4f}")
        
        print("\n📝 LEGENDA:")
        print("🔴 Alta importância (> 0.01) - Remoção causa queda significativa")
        print("🟡 Média importância (0.005-0.01) - Remoção causa queda moderada") 
        print("🟢 Baixa importância (< 0.005) - Remoção causa pouca ou nenhuma queda")
        
        return sorted_importance

    def plot_feature_importance(self):
        if hasattr(self, '_feature_importance_cache'):
            feature_importance = self._feature_importance_cache
        else:
            feature_importance = self.calculate_feature_importance()
            self._feature_importance_cache = feature_importance
        
        if feature_importance is None:
            return
        
        features, importances = zip(*feature_importance)
        
        top_n = min(15, len(features))
        top_features = features[:top_n]
        top_importances = importances[:top_n]
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if imp > 0.01 else 'orange' if imp > 0.005 else 'green' for imp in top_importances]
        bars = plt.barh(range(len(top_features)), top_importances, color=colors, alpha=0.7)
        
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importância (Queda na Acurácia)')
        plt.title(f'Top {top_n} Features Mais Importantes - KNN\n(Baseado na Análise de Sensibilidade)')
        plt.gca().invert_yaxis()
        
        for i, (bar, value) in enumerate(zip(bars, top_importances)):
            plt.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2, 
                    f'{value:+.4f}', ha='left', va='center', fontsize=8)
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.5, label='Alta importância (0.01)')
        plt.axvline(x=0.005, color='orange', linestyle='--', alpha=0.5, label='Média importância (0.005)')
        
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfico de importância das features salvo em: {self.figures_path}/feature_importance.png")

    def save_model(self, filename=None):
        if self.model is None:
            print("❌ Nenhum modelo foi treinado ainda.")
            return False
        
        if filename is None:
            filename = f'src/data/models/modelo_knn_treinado.pkl'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            model_data = {
                'model': self.model,
                'model_type': 'knn',
                'best_params': self.best_params,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'results': self.results,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Modelo KNN salvo com sucesso em: {filename}")
            print(f"   • Acurácia: {self.results.get('test_accuracy', 0):.4f}")
            print(f"   • Data do treinamento: {model_data['training_date']}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {e}")
            return False

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.best_params = model_data.get('best_params', None)
            self.feature_names = model_data.get('feature_names', None)
            self.class_names = model_data.get('class_names', ['Empate', 'Mandante', 'Visitante'])
            self.results = model_data.get('results', {})
            
            print(f"📂 Modelo KNN carregado com sucesso de: {filename}")
            print(f"   • Data do treinamento: {model_data.get('training_date', 'Não disponível')}")
            print(f"   • Acurácia: {self.results.get('test_accuracy', 0):.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False

    def predict_from_saved_model(self, model_path, X_new):
        if self.load_model(model_path):
            try:
                predictions = self.model.predict(X_new)
                probabilities = self.model.predict_proba(X_new) if hasattr(self.model, 'predict_proba') else None
                
                print(f"🔮 Predições realizadas com sucesso!")
                print(f"   • {len(predictions)} predições feitas")
                
                # Mostrar distribuição das predições
                unique, counts = np.unique(predictions, return_counts=True)
                for class_idx, count in zip(unique, counts):
                    class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Classe {class_idx}"
                    print(f"   • {class_name}: {count} predições")
                
                return predictions, probabilities
            except Exception as e:
                print(f"❌ Erro ao fazer predições: {e}")
                return None, None
        else:
            return None, None

    def get_model_info(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"\n📋 INFORMAÇÕES DO MODELO KNN ({model_path}):")
            print("=" * 60)
            print(f"• Tipo: KNN (K-Nearest Neighbors)")
            print(f"• Data do treinamento: {model_data.get('training_date', 'Não disponível')}")
            print(f"• Acurácia de treino: {model_data.get('results', {}).get('train_accuracy', 0):.4f}")
            print(f"• Acurácia de teste: {model_data.get('results', {}).get('test_accuracy', 0):.4f}")
            print(f"• Precisão: {model_data.get('results', {}).get('precision', 0):.4f}")
            print(f"• Recall: {model_data.get('results', {}).get('recall', 0):.4f}")
            print(f"• F1-Score: {model_data.get('results', {}).get('f1_score', 0):.4f}")
            
            if 'best_params' in model_data and model_data['best_params']:
                print(f"• Melhores hiperparâmetros:")
                for param, value in model_data['best_params'].items():
                    print(f"  - {param}: {value}")
            
            if 'feature_names' in model_data and model_data['feature_names']:
                print(f"• Número de features: {len(model_data['feature_names'])}")
            
            return model_data
            
        except Exception as e:
            print(f"❌ Erro ao ler informações do modelo: {e}")
            return None

    def generate_all_figures(self):
        self.create_figures_directory()
        
        print("\n📊 Gerando figuras estatísticas...")
        
        figures = [
            ("Matriz de confusão", self.plot_confusion_matrix),
            ("Curva de validação", self.plot_validation_curve),
            ("Comparação de métricas", self.plot_metrics_comparison),
            ("Melhores hiperparâmetros", self.plot_hyperparameter_importance),
            ("Distribuição das classes", self.plot_class_distribution),
            ("Importância das features", self.plot_feature_importance),
        ]
        
        for i, (desc, func) in enumerate(figures):
            progress_bar(i + 1, len(figures), f"Gerando {desc}")
            func()
            time.sleep(0.3)
        
        self.plot_kmeans_clusters(self.X_train, self.y_train, self.feature_names, f'{self.figures_path}/kmeans_clusters.png')
        self.plot_cluster_distribution(self.X_train, f'{self.figures_path}/kmeans_cluster_distribution.png')
        
        print(f"\n✅ Todas as figuras foram salvas em: {self.figures_path}")

def main():
    print("🚀 INICIANDO ANÁLISE KNN")
    print("=" * 50)
    
    predictor = KNNGamePredictor()

    if not predictor.load_data():
        print("❌ Falha ao carregar dados")
        return

    print("✅ Dados carregados com sucesso")

    try:
        predictor.optimize_hyperparameters()
    except Exception as e:
        print(f"⚠️ Erro na otimização: {e}. Usando parâmetros padrão.")

    print("🎯 Treinando modelo final...")
    predictor.train_model()
    results = predictor.evaluate_model()
    
    print(f"\n🏆 Acurácia final: {results['test_accuracy']:.4f}")
    
    print("\n💾 Salvando modelo treinado...")
    predictor.save_model()
    
    predictor.generate_all_figures()
    
    print("\n✅ Análise KNN concluída!")
    return predictor


if __name__ == "__main__":
    main()
