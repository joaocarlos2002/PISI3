import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import time

class LogisticRegressionGamePredictor:
    def __init__(self, data_path='src/data/data-aprendizado/dados_consolidados.pkl'):
        self.data_path = data_path
        self.model = None
        self.best_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.label_encoder = None
        self.results = {}
        self.figures_path = 'src/data/figuras/regressao'
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
            self.label_encoder = data.get('label_encoder', None)
            return True
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False

    def optimize_hyperparameters(self, cv_folds=5, scoring='accuracy'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*max_iter.*")
            
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [10000],
                'tol': [1e-4]
            }
            grid = GridSearchCV(
                LogisticRegression(random_state=42), 
                param_grid, 
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42), 
                scoring=scoring, 
                n_jobs=-1, 
                verbose=0
            )
            grid.fit(self.X_train, self.y_train)
            self.best_params = grid.best_params_
            
            self.model = LogisticRegression(
                **self.best_params,
                random_state=42
            )
            return grid.best_score_

    def train_model(self, **kwargs):
        if not self.model:
            default_params = {
                'max_iter': 10000,
                'random_state': 42,
                'tol': 1e-4,
                'solver': 'lbfgs',
                'C': 1.0
            }
            default_params.update(kwargs or {})
            self.model = LogisticRegression(**default_params)
        self.model.fit(self.X_train, self.y_train)

    def check_convergence(self):
        if hasattr(self.model, 'n_iter_'):
            if isinstance(self.model.n_iter_, np.ndarray):
                max_iter_reached = np.any(self.model.n_iter_ >= self.model.max_iter)
            else:
                max_iter_reached = self.model.n_iter_ >= self.model.max_iter
            
            if max_iter_reached:
                print("⚠️  AVISO: O modelo pode não ter convergido completamente.")
                print(f"   Iterações utilizadas: {self.model.n_iter_}")
                print(f"   Máximo de iterações: {self.model.max_iter}")
                print("   Considere aumentar max_iter ou ajustar a tolerância.")
            else:
                print("✅ Modelo convergiu com sucesso!")
                print(f"   Iterações utilizadas: {self.model.n_iter_}")
        return True

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
        print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
        print(classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoder.inverse_transform([0, 1, 2])
        ))
        return self.results

    def show_coefficients(self):
        print("\n🔍 ANÁLISE DOS PESOS DAS VARIÁVEIS (REGRESSÃO LOGÍSTICA):")
        print("=" * 70)
        
        coef_df = pd.DataFrame(self.model.coef_, columns=self.feature_names)
        coef_df.index = self.label_encoder.inverse_transform(self.model.classes_)
        
        print("\n📊 TABELA COMPLETA DE COEFICIENTES:")
        print("-" * 70)
        print(coef_df.round(4).T.to_string())
        
        print(f"\n🎯 RANKING DE IMPORTÂNCIA DAS VARIÁVEIS:")
        print("-" * 70)
        
        coef_abs = np.abs(coef_df.values)
        feature_importance = np.mean(coef_abs, axis=0)
        
        importance_ranking = list(zip(self.feature_names, feature_importance))
        importance_ranking.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(importance_ranking, 1):
            if importance > 0.5:
                impact = "🔴"
                level = "ALTA"
            elif importance > 0.2:
                impact = "🟡"
                level = "MÉDIA"
            else:
                impact = "🟢"
                level = "BAIXA"
            
            print(f"{i:2d}. {impact} {feature:<35} | Importância: {importance:.4f} ({level})")
        
        print("\n📝 INTERPRETAÇÃO:")
        print("• Valores POSITIVOS: aumentam a probabilidade da classe")
        print("• Valores NEGATIVOS: diminuem a probabilidade da classe")
        print("• Magnitude maior = maior influência na decisão")
        
        print("\n📈 LEGENDA DE IMPORTÂNCIA:")
        print("🔴 ALTA (> 0.5) - Variável muito influente")
        print("🟡 MÉDIA (0.2-0.5) - Variável moderadamente influente") 
        print("🟢 BAIXA (< 0.2) - Variável pouco influente")
        
        return coef_df

    def create_figures_directory(self):
        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path)

    def plot_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                   xticklabels=self.label_encoder.inverse_transform([0, 1, 2]),
                   yticklabels=self.label_encoder.inverse_transform([0, 1, 2]))
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão - Regressão Logística')
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_coefficients_heatmap(self):
        coef_df = pd.DataFrame(self.model.coef_, columns=self.feature_names)
        coef_df.index = self.label_encoder.inverse_transform(self.model.classes_)
        plt.figure(figsize=(15, 8))
        sns.heatmap(coef_df, annot=False, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Coeficiente'})
        plt.title('Heatmap dos Coeficientes - Regressão Logística')
        plt.xlabel('Features')
        plt.ylabel('Classes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/coefficients_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, top_n=15):
        coef_abs = np.abs(self.model.coef_)
        feature_importance = np.mean(coef_abs, axis=0)
        importance_df = pd.Series(feature_importance, index=self.feature_names)
        top_features = importance_df.sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features.values, color='lightcoral')
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Importância (|Coeficiente| médio)')
        plt.title(f'Top {top_n} Features Mais Importantes - Regressão Logística')
        plt.gca().invert_yaxis()
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left', va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_validation_curve_C(self):
        C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        penalty = self.best_params.get('penalty', 'l2') if self.best_params else 'l2'
        solver = self.best_params.get('solver', 'liblinear') if self.best_params else 'liblinear'
        train_scores, val_scores = validation_curve(
            LogisticRegression(penalty=penalty, solver=solver, max_iter=10000, random_state=42),
            self.X_train, self.y_train, 
            param_name='C', param_range=C_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        plt.figure(figsize=(10, 6))
        plt.semilogx(C_range, train_mean, 'o-', color='blue', label='Treino')
        plt.fill_between(C_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.semilogx(C_range, val_mean, 'o-', color='red', label='Validação')
        plt.fill_between(C_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        if self.best_params and 'C' in self.best_params:
            best_C = self.best_params['C']
            plt.axvline(x=best_C, color='green', 
                       linestyle='--', label=f'Melhor C = {best_C}')
        plt.xlabel('Parâmetro de Regularização (C)')
        plt.ylabel('Acurácia')
        plt.title('Curva de Validação - Regressão Logística')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/validation_curve_C.png', dpi=300, bbox_inches='tight')
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
        plt.title('Métricas de Performance - Regressão Logística')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.figures_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_hyperparameters(self):
        if not self.best_params:
            default_params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'max_iter': 10000
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
        plt.title(f'Melhores Hiperparâmetros - Regressão Logística{title_suffix}')
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
        print("Gerando figuras estatísticas para Regressão Logística...")
        self.plot_confusion_matrix()
        print("✓ Matriz de confusão salva")
        self.plot_coefficients_heatmap()
        print("✓ Heatmap dos coeficientes salvo")
        self.plot_feature_importance()
        print("✓ Importância das features salva")
        self.plot_validation_curve_C()
        print("✓ Curva de validação salva")
        self.plot_metrics_comparison()
        print("✓ Comparação de métricas salva")
        self.plot_hyperparameters()
        print("✓ Melhores hiperparâmetros salvos")
        self.plot_class_distribution()
        print("✓ Distribuição das classes salva")
        print(f"\nTodas as figuras foram salvas em: {self.figures_path}")

    def save_model(self, filename=None):
        if self.model is None:
            print("❌ Nenhum modelo foi treinado ainda.")
            return False
        
        if filename is None:
            filename = f'src/data/models/modelo_regressao_treinado.pkl'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            model_data = {
                'model': self.model,
                'model_type': 'logistic_regression',
                'best_params': self.best_params,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'label_encoder': self.label_encoder,
                'results': self.results,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Modelo Regressão Logística salvo com sucesso em: {filename}")
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
            self.label_encoder = model_data.get('label_encoder', None)
            self.results = model_data.get('results', {})
            
            print(f"📂 Modelo Regressão Logística carregado com sucesso de: {filename}")
            print(f"   • Data do treinamento: {model_data.get('training_date', 'Não disponível')}")
            print(f"   • Acurácia: {self.results.get('test_accuracy', 0):.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False

    def predict_from_saved_model(self, model_path, X_new):
        """Carrega um modelo salvo e faz predições em novos dados"""
        if self.load_model(model_path):
            try:
                predictions = self.model.predict(X_new)
                probabilities = self.model.predict_proba(X_new) if hasattr(self.model, 'predict_proba') else None
                
                print(f"🔮 Predições realizadas com sucesso!")
                print(f"   • {len(predictions)} predições feitas")
                
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
        """Mostra informações sobre um modelo salvo sem carregá-lo completamente"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"\n📋 INFORMAÇÕES DO MODELO REGRESSÃO LOGÍSTICA ({model_path}):")
            print("=" * 70)
            print(f"• Tipo: Regressão Logística")
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

def main():
    predictor = LogisticRegressionGamePredictor()
    if not predictor.load_data():
        return
    
    print("Iniciando otimização de hiperparâmetros...")
    try:
        best_score = predictor.optimize_hyperparameters()
        print(f"Melhor score na validação cruzada: {best_score:.4f}")
    except Exception as e:
        print(f"Erro na otimização de hiperparâmetros: {e}")
        print("Usando parâmetros padrão...")
    
    print("Treinando modelo...")
    predictor.train_model()
    
    predictor.check_convergence()
    
    results = predictor.evaluate_model()
    predictor.show_coefficients()
    print(f"\nAcurácia final: {results['test_accuracy']:.4f}")
    print("\n💾 Salvando modelo treinado...")
    predictor.save_model()
    
    predictor.generate_all_figures()
    return predictor

if __name__ == "__main__":
    main()