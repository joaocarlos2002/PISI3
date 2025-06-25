import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

class TreeGamePredictor:
    def __init__(self, data_path='src/data/data-aprendizado/dados_consolidados.pkl', model_type='tree'):
        self.data_path = data_path
        self.model_type = model_type 
        self.model = None
        self.best_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}
        self.figures_path = f'src/data/figuras/arvore'
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

    def optimize_hyperparameters(self):
        model = DecisionTreeClassifier() if self.model_type == 'tree' else RandomForestClassifier()
        param_grid = {
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        if self.model_type == 'forest':
            param_grid['n_estimators'] = [50, 100, 200]

        grid = GridSearchCV(model, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

    def train_model(self):
        if not self.model:
            if self.model_type == 'tree':
                self.model = DecisionTreeClassifier()
            else:
                self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        self.results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
        }
        print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
        print(classification_report(self.y_test, y_pred, target_names=['Empate', 'Mandante', 'Visitante']))
        return self.results

    def show_feature_importance(self, top_n=10):
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
            top_features = importances.sort_values(ascending=False).head(top_n)
            print(f"\n🌟 Top {top_n} features mais importantes:")
            print(top_features.to_string())
        else:
            print("Este modelo não suporta extração de importância de features.")

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
        model_name = 'Árvore de Decisão' if self.model_type == 'tree' else 'Floresta Aleatória'
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.xlabel('Predição')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/confusion_matrix_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, top_n=15):
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
            top_features = importances.sort_values(ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            bars = plt.barh(range(len(top_features)), top_features.values, color='lightcoral')
            plt.yticks(range(len(top_features)), top_features.index)
            plt.xlabel('Importância')
            model_name = 'Árvore de Decisão' if self.model_type == 'tree' else 'Floresta Aleatória'
            plt.title(f'Top {top_n} Features Mais Importantes - {model_name}')
            plt.gca().invert_yaxis()
            
            for i, (bar, value) in enumerate(zip(bars, top_features.values)):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{self.figures_path}/feature_importance_{self.model_type}.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_tree_visualization(self):
        if self.model_type == 'tree' and hasattr(self.model, 'tree_'):
            plt.figure(figsize=(20, 10))
            plot_tree(self.model, 
                     feature_names=self.feature_names,
                     class_names=self.class_names,
                     filled=True,
                     rounded=True,
                     fontsize=8,
                     max_depth=3)
            plt.title('Árvore de Decisão (Primeiros 3 níveis)')
            plt.tight_layout()
            plt.savefig(f'{self.figures_path}/tree_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_validation_curve_depth(self):
        if self.model_type == 'tree':
            model_class = DecisionTreeClassifier
            title = 'Árvore de Decisão'
        else:
            model_class = RandomForestClassifier
            title = 'Floresta Aleatória'
        
        depth_range = range(1, 21)
        train_scores, val_scores = validation_curve(
            model_class(random_state=42),
            self.X_train, self.y_train, 
            param_name='max_depth', param_range=depth_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(depth_range, train_mean, 'o-', color='blue', label='Treino')
        plt.fill_between(depth_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(depth_range, val_mean, 'o-', color='red', label='Validação')
        plt.fill_between(depth_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        if self.best_params and 'max_depth' in self.best_params:
            best_depth = self.best_params['max_depth']
            if best_depth:
                plt.axvline(x=best_depth, color='green', 
                           linestyle='--', label=f'Melhor Profundidade = {best_depth}')
        
        plt.xlabel('Profundidade Máxima')
        plt.ylabel('Acurácia')
        plt.title(f'Curva de Validação - {title}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/validation_curve_depth_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metrics_comparison(self):
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [self.results[metric] for metric in metrics if metric in self.results]
        metric_labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_labels, values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1.1)
        plt.ylabel('Score')
        model_name = 'Árvore de Decisão' if self.model_type == 'tree' else 'Floresta Aleatória'
        plt.title(f'Métricas de Performance - {model_name}')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/metrics_comparison_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_hyperparameters(self):
        if not self.best_params:
            return
            
        relevant_params = {}
        for key, value in self.best_params.items():
            if key in ['max_depth', 'min_samples_split', 'criterion', 'n_estimators']:
                relevant_params[key] = str(value)
        
        if not relevant_params:
            return
            
        param_names = list(relevant_params.keys())
        param_values = list(relevant_params.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(param_names, range(len(param_names)), color='lightblue')
        
        for i, (bar, value) in enumerate(zip(bars, param_values)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    value, ha='left', va='center', fontweight='bold')
        
        plt.xlabel('Ordem de Importância')
        model_name = 'Árvore de Decisão' if self.model_type == 'tree' else 'Floresta Aleatória'
        plt.title(f'Melhores Hiperparâmetros - {model_name}')
        plt.xlim(0, len(param_names) + 1)
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/best_hyperparameters_{self.model_type}.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(f'{self.figures_path}/class_distribution_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_figures(self):
        self.create_figures_directory()
        
        model_name = 'Árvore de Decisão' if self.model_type == 'tree' else 'Floresta Aleatória'
        print(f"Gerando figuras estatísticas para {model_name}...")
        
        self.plot_confusion_matrix()
        print("✓ Matriz de confusão salva")
        
        self.plot_feature_importance()
        print("✓ Importância das features salva")
        
        if self.model_type == 'tree':
            self.plot_tree_visualization()
            print("✓ Visualização da árvore salva")
        
        self.plot_validation_curve_depth()
        print("✓ Curva de validação salva")
        
        self.plot_metrics_comparison()
        print("✓ Comparação de métricas salva")
        
        self.plot_hyperparameters()
        print("✓ Melhores hiperparâmetros salvos")
        
        self.plot_class_distribution()
        print("✓ Distribuição das classes salva")
        
        print(f"\nTodas as figuras foram salvas em: {self.figures_path}")

def main_tree(model_type='tree'):
    print(f"🌲 Iniciando modelo: {'Árvore de Decisão' if model_type == 'tree' else 'Floresta Aleatória'}")
    predictor = TreeGamePredictor(model_type=model_type)

    if not predictor.load_data():
        return

    predictor.optimize_hyperparameters()
    predictor.train_model()
    results = predictor.evaluate_model()
    predictor.show_feature_importance()
    
    print(f"\n🎯 Acurácia final: {results['accuracy']:.4f}")
    
    predictor.generate_all_figures()
    
    return predictor


if __name__ == "__main__":
    main_tree(model_type='forest')
