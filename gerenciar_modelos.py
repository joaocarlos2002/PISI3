import pickle
import pandas as pd
import numpy as np
import os
import sys


sys.path.append('src')

from aprendizado.regressao.base_games import LogisticRegressionGamePredictor
from aprendizado.knn.base_games import KNNGamePredictor
from aprendizado.arvore.base_games import TreeGamePredictor

def listar_modelos_salvos():
    models_dir = 'src/data/models'
    
    if not os.path.exists(models_dir):
        print("❌ Diretório de modelos não encontrado.")
        return []
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not models:
        print("❌ Nenhum modelo salvo encontrado.")
        return []
    
    print("\n📂 MODELOS SALVOS DISPONÍVEIS:")
    print("=" * 50)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    return [os.path.join(models_dir, model) for model in models]

def mostrar_info_modelo(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model_type = model_data.get('model_type', 'desconhecido')
        
        print(f"\n📋 INFORMAÇÕES DETALHADAS DO MODELO:")
        print("=" * 60)
        print(f"📄 Arquivo: {os.path.basename(model_path)}")
        print(f"🧠 Tipo: {model_type}")
        print(f"📅 Data do treinamento: {model_data.get('training_date', 'Não disponível')}")
        
        results = model_data.get('results', {})
        if results:
            print(f"\n📊 MÉTRICAS DE PERFORMANCE:")
            print(f"   • Acurácia de treino: {results.get('train_accuracy', results.get('accuracy', 0)):.4f}")
            print(f"   • Acurácia de teste: {results.get('test_accuracy', results.get('accuracy', 0)):.4f}")
            print(f"   • Precisão: {results.get('precision', 0):.4f}")
            print(f"   • Recall: {results.get('recall', 0):.4f}")
            print(f"   • F1-Score: {results.get('f1_score', 0):.4f}")
        
        best_params = model_data.get('best_params', {})
        if best_params:
            print(f"\n⚙️ HIPERPARÂMETROS OTIMIZADOS:")
            for param, value in best_params.items():
                print(f"   • {param}: {value}")
        
        feature_names = model_data.get('feature_names', [])
        if feature_names:
            print(f"\n🔍 FEATURES UTILIZADAS:")
            print(f"   • Total: {len(feature_names)} variáveis")
            print(f"   • Primeiras 5: {feature_names[:5]}")
            if len(feature_names) > 5:
                print(f"   • ... e mais {len(feature_names) - 5} variáveis")
        
        return model_data
        
    except Exception as e:
        print(f"❌ Erro ao ler modelo: {e}")
        return None

def carregar_e_testar_modelo(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model_type = model_data.get('model_type', 'desconhecido')
        
        if model_type == 'logistic_regression':
            predictor = LogisticRegressionGamePredictor()
        elif model_type == 'knn':
            predictor = KNNGamePredictor()
        elif model_type in ['tree', 'forest']:
            predictor = TreeGamePredictor(model_type=model_type)
        else:
            print(f"❌ Tipo de modelo não reconhecido: {model_type}")
            return None
        
        if predictor.load_model(model_path):
            print("\n✅ Modelo carregado com sucesso!")
            
            if predictor.load_data():
                print("✅ Dados de teste carregados!")
                
                sample_size = min(10, len(predictor.X_test))
                X_sample = predictor.X_test[:sample_size]
                y_sample = predictor.y_test[:sample_size]
                
                predictions = predictor.model.predict(X_sample)
                
                print(f"\n🔮 EXEMPLO DE PREDIÇÕES (primeiras {sample_size} amostras):")
                print("─" * 50)
                print(f"{'#':<3} {'Real':<10} {'Predito':<10} {'Correto':<8}")
                print("─" * 50)
                
                correct = 0
                for i in range(sample_size):
                    real_class = predictor.class_names[y_sample[i]]
                    pred_class = predictor.class_names[predictions[i]]
                    is_correct = "✅" if y_sample[i] == predictions[i] else "❌"
                    if y_sample[i] == predictions[i]:
                        correct += 1
                    
                    print(f"{i+1:<3} {real_class:<10} {pred_class:<10} {is_correct}")
                
                accuracy = correct / sample_size
                print("─" * 50)
                print(f"📊 Acurácia na amostra: {accuracy:.2%} ({correct}/{sample_size})")
                
                return predictor
            
        return None
        
    except Exception as e:
        print(f"❌ Erro ao carregar e testar modelo: {e}")
        return None

def comparar_modelos():
    models = listar_modelos_salvos()
    
    if not models:
        return
    
    print("\n📊 COMPARAÇÃO DE MODELOS:")
    print("=" * 80)
    print(f"{'MODELO':<25} {'TIPO':<15} {'ACURÁCIA':<10} {'PRECISÃO':<10} {'F1-SCORE':<10}")
    print("─" * 80)
    
    model_performances = []
    
    for model_path in models:
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model_name = os.path.basename(model_path).replace('.pkl', '')
            model_type = model_data.get('model_type', 'desconhecido')
            results = model_data.get('results', {})
            
            accuracy = results.get('test_accuracy', results.get('accuracy', 0))
            precision = results.get('precision', 0)
            f1_score = results.get('f1_score', 0)
            
            print(f"{model_name:<25} {model_type:<15} {accuracy:<10.4f} {precision:<10.4f} {f1_score:<10.4f}")
            
            model_performances.append({
                'name': model_name,
                'type': model_type,
                'accuracy': accuracy,
                'precision': precision,
                'f1_score': f1_score
            })
            
        except Exception as e:
            print(f"❌ Erro ao ler {model_path}: {e}")
    
    if model_performances:
        best_model = max(model_performances, key=lambda x: x['accuracy'])
        print("─" * 80)
        print(f"🏆 MELHOR MODELO: {best_model['name']} (Acurácia: {best_model['accuracy']:.4f})")

def fazer_predicao_personalizada():
    models = listar_modelos_salvos()
    
    if not models:
        return
    
    try:
        idx = int(input(f"\n👉 Escolha um modelo para predição (1-{len(models)}): ")) - 1
        if not (0 <= idx < len(models)):
            print("❌ Índice inválido!")
            return
        
        model_path = models[idx]
        model_data = None
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model_type = model_data.get('model_type', 'desconhecido')
        
        if model_type == 'logistic_regression':
            predictor = LogisticRegressionGamePredictor()
        elif model_type == 'knn':
            predictor = KNNGamePredictor()
        elif model_type in ['tree', 'forest']:
            predictor = TreeGamePredictor(model_type=model_type)
        else:
            print(f"❌ Tipo de modelo não reconhecido: {model_type}")
            return
        
        if predictor.load_model(model_path) and predictor.load_data():
            print(f"\n✅ Modelo {model_type} carregado!")
            
            import random
            sample_idx = random.randint(0, len(predictor.X_test) - 1)
            
            X_sample = predictor.X_test[sample_idx:sample_idx+1]
            y_real = predictor.y_test[sample_idx]
            
            prediction = predictor.model.predict(X_sample)[0]
            probabilities = predictor.model.predict_proba(X_sample)[0]
            
            print(f"\n🎯 PREDIÇÃO PERSONALIZADA:")
            print("─" * 40)
            print(f"Resultado Real: {predictor.class_names[y_real]}")
            print(f"Resultado Predito: {predictor.class_names[prediction]}")
            print(f"Correto: {'✅' if y_real == prediction else '❌'}")
            
            print(f"\n📊 PROBABILIDADES:")
            for i, prob in enumerate(probabilities):
                print(f"   {predictor.class_names[i]}: {prob:.2%}")
            
        else:
            print("❌ Erro ao carregar modelo ou dados!")
            
    except ValueError:
        print("❌ Por favor, digite um número válido!")
    except Exception as e:
        print(f"❌ Erro: {e}")

def main():
    print("🚀 SISTEMA DE GERENCIAMENTO DE MODELOS TREINADOS")
    print("=" * 60)
    
    while True:
        print("\n📋 OPÇÕES DISPONÍVEIS:")
        print("1. 📂 Listar modelos salvos")
        print("2. 📋 Mostrar informações de um modelo")
        print("3. 🧪 Testar um modelo com dados")
        print("4. 📊 Comparar todos os modelos")
        print("5. 🎯 Fazer predição personalizada")
        print("6. 🚪 Sair")
        
        try:
            opcao = input("\n👉 Escolha uma opção (1-6): ").strip()
            
            if opcao == '1':
                listar_modelos_salvos()
                
            elif opcao == '2':
                models = listar_modelos_salvos()
                if models:
                    try:
                        idx = int(input(f"\n👉 Escolha um modelo (1-{len(models)}): ")) - 1
                        if 0 <= idx < len(models):
                            mostrar_info_modelo(models[idx])
                        else:
                            print("❌ Índice inválido!")
                    except ValueError:
                        print("❌ Por favor, digite um número válido!")
                        
            elif opcao == '3':
                models = listar_modelos_salvos()
                if models:
                    try:
                        idx = int(input(f"\n👉 Escolha um modelo para testar (1-{len(models)}): ")) - 1
                        if 0 <= idx < len(models):
                            carregar_e_testar_modelo(models[idx])
                        else:
                            print("❌ Índice inválido!")
                    except ValueError:
                        print("❌ Por favor, digite um número válido!")
                        
            elif opcao == '4':
                comparar_modelos()
                
            elif opcao == '5':
                fazer_predicao_personalizada()
                
            elif opcao == '6':
                print("\n👋 Obrigado por usar o sistema!")
                break
                
            else:
                print("❌ Opção inválida! Escolha entre 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Sistema encerrado pelo usuário.")
            break
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()
