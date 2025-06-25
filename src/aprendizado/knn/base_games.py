import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

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

        grid = GridSearchCV(KNeighborsClassifier(), combinations, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42), scoring=scoring, n_jobs=-1, verbose=0)
        grid.fit(self.X_train, self.y_train)
        self.best_params = grid.best_params_
        self.model = grid.best_estimator_
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

    def generate_all_figures(self):
        self.create_figures_directory()
        
        print("Gerando figuras estatísticas...")
        
        self.plot_confusion_matrix()
        print("✓ Matriz de confusão salva")
        
        self.plot_validation_curve()
        print("✓ Curva de validação salva")
        
        self.plot_metrics_comparison()
        print("✓ Comparação de métricas salva")
        
        self.plot_hyperparameter_importance()
        print("✓ Melhores hiperparâmetros salvos")
        
        self.plot_class_distribution()
        print("✓ Distribuição das classes salva")
        
        print(f"\nTodas as figuras foram salvas em: {self.figures_path}")

def main():
    predictor = KNNGamePredictor()

    if not predictor.load_data():
        return

    try:
        predictor.optimize_hyperparameters()
    except Exception:
        pass

    predictor.train_model()
    results = predictor.evaluate_model()
    print(f"Acurácia final: {results['test_accuracy']:.4f}")
    
    predictor.generate_all_figures()
    
    return predictor


if __name__ == "__main__":
    main()
