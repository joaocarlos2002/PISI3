import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

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

    def show_progress_bar(self, current, total, operation="Processando"):
        bar_length = 40
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        percent = 100 * (current / float(total))
        print(f'\r{operation}: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
        if current == total:
            print() 

    def load_data(self):
        print("🔄 Carregando dados...")
        try:
            for i in range(1, 4):
                self.show_progress_bar(i, 3, "Carregando dados")
                time.sleep(0.3)
            
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            self.X_train = data['X_train']
            self.X_test = data['X_test']
            self.y_train = data['y_train']
            self.y_test = data['y_test']
            self.feature_names = data.get('features', None)
            print("✅ Dados carregados com sucesso!")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False

    def optimize_hyperparameters(self):
        print("🔧 Otimizando hiperparâmetros...")
        
        for i in range(1, 4):
            self.show_progress_bar(i, 3, "Preparando grid search")
            time.sleep(0.2)
        
        model = DecisionTreeClassifier() if self.model_type == 'tree' else RandomForestClassifier()
        param_grid = {
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        if self.model_type == 'forest':
            param_grid['n_estimators'] = [50, 100, 200]

        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        print(f"🔍 Testando {total_combinations} combinações de hiperparâmetros...")
        
        # Simular progresso do grid search
        for i in range(1, 11):
            self.show_progress_bar(i, 10, "Grid Search")
            time.sleep(0.5)
        
        grid = GridSearchCV(model, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        print("✅ Otimização concluída!")

    def train_model(self):
        print("🚀 Treinando modelo...")
        if not self.model:
            if self.model_type == 'tree':
                self.model = DecisionTreeClassifier()
            else:
                self.model = RandomForestClassifier()
    
        for i in range(1, 6):
            self.show_progress_bar(i, 5, "Treinamento")
            time.sleep(0.3)
            
        self.model.fit(self.X_train, self.y_train)
        print("✅ Modelo treinado com sucesso!")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        self.results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
        }
        print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
        print(classification_report(self.y_test, y_pred, target_names=['Empate', 'Mandante', 'Visitante']))
        return self.results

    def show_feature_importance(self, top_n=15):
        if hasattr(self.model, 'feature_importances_'):
            model_name = 'Árvore de Decisão' if self.model_type == 'tree' else 'Floresta Aleatória'
            print(f"\n🌳 ANÁLISE DE IMPORTÂNCIA DAS FEATURES ({model_name.upper()}):")
            print("=" * 80)
            
            importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
            all_features = importances.sort_values(ascending=False)
            
            print(f"\n📊 RANKING DE IMPORTÂNCIA DAS VARIÁVEIS ({model_name}):")
            print("─" * 80)
            print(f"{'#':<3} {'📈':<2} {'VARIÁVEL':<40} {'IMPORTÂNCIA':<12} {'%':<8} {'NÍVEL':<8}")
            print("─" * 80)
            
            for i, (feature, importance) in enumerate(all_features.items(), 1):
                if importance > 0.15:
                    icon, level, color = "🔴", "ALTA", "\033[91m"
                elif importance > 0.05:
                    icon, level, color = "🟡", "MÉDIA", "\033[93m"
                else:
                    icon, level, color = "🟢", "BAIXA", "\033[92m"
                
                percentage = importance * 100
                reset_color = "\033[0m"
                print(f"{i:2d}  {icon}  {color}{feature:<40}{reset_color} {importance:<12.6f} {percentage:6.2f}%  {level}")
            
            print("─" * 80)
            print(f"\n ESTATÍSTICAS RESUMIDAS:")
            print(f"   • Total de variáveis analisadas: {len(all_features)}")
            print(f"   • Variável mais importante: '{all_features.index[0]}' ({all_features.iloc[0]:.4f})")
            print(f"   • Importância média: {all_features.mean():.4f}")
            print(f"   • Distribuição de níveis:")
            
            alta_count = (all_features > 0.15).sum()
            media_count = ((all_features > 0.05) & (all_features <= 0.15)).sum()
            baixa_count = (all_features <= 0.05).sum()
            
            print(f"     🔴 ALTA (>15%): {alta_count} variáveis")
            print(f"     🟡 MÉDIA (5-15%): {media_count} variáveis") 
            print(f"     🟢 BAIXA (<5%): {baixa_count} variáveis")
            
            print(f"\n🎯 TOP {top_n} VARIÁVEIS MAIS IMPORTANTES:")
            print("─" * 60)
            top_features = all_features.head(top_n)
            for i, (feature, importance) in enumerate(top_features.items(), 1):
                bar_length = int(30 * importance / all_features.max())
                bar = "█" * bar_length + "░" * (30 - bar_length)
                print(f"{i:2d}. {feature:<35} |{bar}| {importance:.4f}")
            
            print(f"\n💡 INTERPRETAÇÃO ({model_name}):")
            print("   • Valores representam a redução média de impureza (Gini/Entropia)")
            print("   • Soma total deve ser ≈ 1.0 (normalizado)")
            print("   • Features com maior importância têm mais poder preditivo")
            
            return all_features
        else:
            print("❌ Este modelo não suporta extração de importância de features.")
            return None

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
        print(f"\n📊 Gerando figuras estatísticas para {model_name}...")
        
        figures_to_generate = [
            ("Matriz de confusão", self.plot_confusion_matrix),
            ("Importância das features", self.plot_feature_importance),
            ("Curva de validação", self.plot_validation_curve_depth),
            ("Comparação de métricas", self.plot_metrics_comparison),
            ("Hiperparâmetros", self.plot_hyperparameters),
            ("Distribuição das classes", self.plot_class_distribution)
        ]
        
        if self.model_type == 'tree':
            figures_to_generate.insert(2, ("Visualização da árvore", self.plot_tree_visualization))
        
        for i, (name, func) in enumerate(figures_to_generate, 1):
            self.show_progress_bar(i, len(figures_to_generate), "Gerando figuras")
            func()
            time.sleep(0.2)
        
        print(f"\n✅ Todas as figuras foram salvas em: {self.figures_path}")

    def save_model(self, filename=None):
        if self.model is None:
            print("❌ Nenhum modelo foi treinado ainda.")
            return False
        
        if filename is None:
            model_name = 'tree' if self.model_type == 'tree' else 'forest'
            filename = f'src/data/models/modelo_{model_name}_treinado.pkl'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'best_params': self.best_params,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'results': self.results,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            model_name = 'Árvore de Decisão' if self.model_type == 'tree' else 'Floresta Aleatória'
            print(f"💾 Modelo {model_name} salvo com sucesso em: {filename}")
            print(f"   • Acurácia: {self.results.get('accuracy', 0):.4f}")
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
            self.model_type = model_data.get('model_type', self.model_type)
            self.best_params = model_data.get('best_params', None)
            self.feature_names = model_data.get('feature_names', None)
            self.class_names = model_data.get('class_names', ['Empate', 'Mandante', 'Visitante'])
            self.results = model_data.get('results', {})
            
            model_name = 'Árvore de Decisão' if self.model_type == 'tree' else 'Floresta Aleatória'
            print(f"📂 Modelo {model_name} carregado com sucesso de: {filename}")
            print(f"   • Data do treinamento: {model_data.get('training_date', 'Não disponível')}")
            print(f"   • Acurácia: {self.results.get('accuracy', 0):.4f}")
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
            
            model_name = 'Árvore de Decisão' if model_data.get('model_type') == 'tree' else 'Floresta Aleatória'
            print(f"\n📋 INFORMAÇÕES DO MODELO ({model_path}):")
            print("=" * 60)
            print(f"• Tipo: {model_name}")
            print(f"• Data do treinamento: {model_data.get('training_date', 'Não disponível')}")
            print(f"• Acurácia: {model_data.get('results', {}).get('accuracy', 0):.4f}")
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
def main_tree(model_type='tree'):
    model_name = 'Árvore de Decisão' if model_type == 'tree' else 'Floresta Aleatória'
    print(f"\n🌳 Iniciando análise com {model_name}")
    print("=" * 50)
    
    predictor = TreeGamePredictor(model_type=model_type)
    
    if not predictor.load_data():
        return
    

    predictor.optimize_hyperparameters()
    

    predictor.train_model()
    

    print("📈 Avaliando modelo...")
    results = predictor.evaluate_model()
    

    print("\n🔍 Analisando importância das features...")
    predictor.show_feature_importance()
    
    print(f"\n🎯 Acurácia final: {results['accuracy']:.4f}")
    
    print("\n💾 Salvando modelo treinado...")
    predictor.save_model()
    
    predictor.generate_all_figures()
    
    print(f"\n✅ Análise de {model_name} concluída!")
    print("=" * 50)
    return predictor

if __name__ == "__main__":
#    main_tree(model_type='tree')
    main_tree(model_type='forest')
    if os.path.exists('src/data/figuras/arvore'):
        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt
        predictor = TreeGamePredictor(model_type='tree')
        if predictor.load_data():
            predictor.optimize_hyperparameters()
            predictor.train_model()
            if hasattr(predictor.model, 'tree_'):
                plt.figure(figsize=(20, 10))
                plot_tree(predictor.model, 
                          feature_names=predictor.feature_names,
                          class_names=predictor.class_names,
                          filled=True,
                          rounded=True,
                          fontsize=8,
                          max_depth=15)
                plt.title('Árvore de Decisão')
                plt.tight_layout()
                plt.savefig('src/data/figuras/arvore/tree_visualization.png', dpi=300, bbox_inches='tight')
                plt.close()
