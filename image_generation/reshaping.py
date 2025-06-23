# Imports originais
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import numpy as np
import io
from pyts.image import GramianAngularField, RecurrencePlot
import os 
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
import gc 

# --- Parte 1: Carregamento e Pré-processamento (praticamente inalterada) ---
# Esta parte do seu código já é eficiente para carregar e transformar os dados.

print("Iniciando o carregamento e pré-processamento...")
domains = pd.read_csv("../datasets/dataset.csv")
domain_urls = domains['name']
labels = domains['malicious'].values # Usar .values para obter um array numpy para estratificação

dns = domains.drop(columns=['name','malicious'])

scaler = MinMaxScaler()
dns = scaler.fit_transform(dns)
print("DNS Features Scaled")

vectorizer = TfidfVectorizer(analyzer="char", sublinear_tf=True,lowercase=False, ngram_range=(3,3), max_features=1024)
features_np = []
batch_size = 2048

for i in range(0,len(domains), batch_size):
    batch_domains = domain_urls[i:i + batch_size]
    batch_features_gpu = vectorizer.fit_transform(batch_domains).toarray()
    features_np.append(batch_features_gpu)
    gc.collect() 
    print(f"Processado lote TF-IDF {i // batch_size + 1}/{(len(domain_urls) + batch_size - 1) // batch_size}")
    
X = np.concatenate(features_np, axis=0) 
del features_np
gc.collect()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print("Domain Vectorized Scaled")
data = np.hstack([X,dns])
del X, dns
gc.collect()

pca = PCA(n_components=64)
X_processed = pca.fit_transform(data)
del data
gc.collect()

print("Dimensions Reduced In-Order to create Images")
print(f"Shape após redução (dataset completo): {X_processed.shape}")
print(f"Total de amostras: {len(X_processed)}")
print(f"Contagem de classes: Benigno={np.sum(labels == 0)}, Malicioso={np.sum(labels == 1)}")

# --- FIM DA PARTE DE PRÉ-PROCESSAMENTO ---


# --- INÍCIO DA LÓGICA DE GERAÇÃO E SALVAMENTO EM BATCH (MODIFICADA) ---
# A seção de criação de subset foi removida.

image_reshapes = {
    "GASF": GramianAngularField(method = "summation"),
    "GADF": GramianAngularField(method = "difference"),
    "RPLOT": RecurrencePlot(dimension=1,threshold='point', percentage=20) 
}

states = [0, 100, 1000]
c = 0 # Contador para os diretórios de cada state (ex: GASF0, GASF1, etc)

for state in states:   
    print(f"\n--- INICIANDO EXECUÇÃO COM RANDOM_STATE = {state} ---")

    # PASSO 1: Dividir os ÍNDICES do dataset completo ANTES de gerar imagens.
    # Isso define para qual conjunto (train/val/test) cada amostra irá.
    # `stratify=labels` é crucial para manter a proporção de classes.
    all_indices = np.arange(len(X_processed))
    
    # Primeira divisão para obter o conjunto de treino (70%) e um conjunto temporário (30%)
    train_indices, temp_indices, _, temp_labels = train_test_split(
        all_indices, labels, test_size=0.3, stratify=labels, random_state=state
    )
    
    # Segunda divisão no conjunto temporário para obter validação (15%) e teste (15%)
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=state
    )
    print("Só pra commitar")
    print(f"Divisão do dataset: {len(train_indices)} treino, {len(val_indices)} validação, {len(test_indices)} teste.")

    # PASSO 2: Criar um mapa para consulta rápida: de índice para conjunto
    index_to_set_map = {idx: 'train' for idx in train_indices}
    index_to_set_map.update({idx: 'val' for idx in val_indices})
    index_to_set_map.update({idx: 'test' for idx in test_indices})

    for image_type, transformer in image_reshapes.items():
        print(f"\n> Processando tipo de imagem: {image_type}")
        base_dir = f"../datasets/domain-TFIDV-Full/{str(image_type)}{c}"

        # PASSO 3: Criar diretórios e contadores de arquivo para esta execução
        file_counters = {}
        for subdir in ["train", "val", "test"]:
            os.makedirs(os.path.join(base_dir, subdir, "benign"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, subdir, "malicious"), exist_ok=True)
            file_counters[subdir] = {"benign": 0, "malicious": 0}

        # PASSO 4: Processar em lotes e salvar imagens diretamente no disco
        batch_size_imgs = 256 # Tamanho do lote para geração de imagens. Ajustável.

        for i in range(0, len(X_processed), batch_size_imgs):
            # Índices originais do lote atual
            batch_original_indices = all_indices[i : i + batch_size_imgs]
            
            # Dados e rótulos correspondentes a este lote
            batch_data = X_processed[batch_original_indices]
            batch_labels = labels[batch_original_indices]
            
            # Gera as imagens APENAS para o lote atual
            generated_images = transformer.fit_transform(batch_data)
            
            # Itera sobre as imagens geradas no lote para salvá-las
            for j, original_idx in enumerate(batch_original_indices):
                image_to_save = generated_images[j]
                label = batch_labels[j]
                
                # Determina onde salvar usando o mapa
                dataset_name = index_to_set_map[original_idx]  # 'train', 'val', ou 'test'
                class_name = "malicious" if label == 1 else "benign"
                
                # Define cores e nome do arquivo
                colors = "binary" if image_type == "RPLOT" else "rainbow"
                count = file_counters[dataset_name][class_name]
                file_path = os.path.join(base_dir, dataset_name, class_name, f"{class_name}_{count}.png")
                
                # Salva a imagem e incrementa o contador
                plt.imsave(file_path, image_to_save, cmap=colors)
                file_counters[dataset_name][class_name] += 1

            print(f"  Lote de imagens {image_type} {i // batch_size_imgs + 1}/{(len(X_processed) + batch_size_imgs - 1) // batch_size_imgs} salvo.")
            gc.collect()
            
        print(f"Imagens {image_type} (state {state}) geradas e salvas com sucesso!")
        
    c += 1

print("\n--- PROCESSO TOTALMENTE CONCLUÍDO ---")