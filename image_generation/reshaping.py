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

def save_images(dataset_name, X_data, y_data, arg):
    for i, (image, label) in enumerate(zip(X_data, y_data)):
        class_name = "malicious" if label == 1 else "benign"
        file_path = os.path.join(
            base_dir, dataset_name, class_name, f"{class_name}_{i}.png"
        )
        if arg == "RPLOT":
            colors = "binary"
        else:
            colors = "rainbow"
        plt.imsave(file_path, image, cmap=colors)
    print(f"Imagens salvas em {base_dir}/{dataset_name}")

domains = pd.read_csv("../datasets/dataset.csv")
domain_urls = domains['name']
labels = domains['malicious']

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
    print(f"Processado lote {i // batch_size + 1}/{(len(domain_urls) + batch_size - 1) // batch_size}")
    
X = np.concatenate(features_np, axis=0) 

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

print("Domain Vectorized Scaled")
data = np.hstack([X,dns])

pca = PCA(n_components=64)
X_processed = pca.fit_transform(data)
gc.collect()

print("Dimensions Reduced In-Order to create Images")
print(f"Shape after feature Reduction: {X_processed[0].shape}")

image_reshapes = {
    "GASF": GramianAngularField(method = "summation"),
    "GADF": GramianAngularField(method = "difference"),
    "RPLOT": RecurrencePlot(dimension=1,threshold='point', percentage=20) 
}

i=0


samples_per_class = 5000
total_subset_size = samples_per_class * 2 # 5000 benignas + 5000 malignas

# --- 3. Separar Índices por Classe ---
# Encontra os índices das amostras benignas e malignas no dataset completo
benign_indices = np.where(labels == 0)[0]
malicious_indices = np.where(labels == 1)[0]

print(f"\nTotal de amostras benignas disponíveis: {len(benign_indices)}")
print(f"Total de amostras malignas disponíveis: {len(malicious_indices)}")

# --- 4. Selecionar Aleatoriamente 5000 Índices de Cada Classe ---
# Verifica se há amostras suficientes para cada classe
if len(benign_indices) < samples_per_class:
    raise ValueError(f"Não há {samples_per_class} amostras benignas suficientes no dataset. Encontradas: {len(benign_indices)}")
if len(malicious_indices) < samples_per_class:
    raise ValueError(f"Não há {samples_per_class} amostras malignas suficientes no dataset. Encontradas: {len(malicious_indices)}")

# Seleciona aleatoriamente 'samples_per_class' índices de cada grupo
selected_benign_indices = np.random.choice(benign_indices, samples_per_class, replace=False)
selected_malicious_indices = np.random.choice(malicious_indices, samples_per_class, replace=False)

# --- 5. Combinar os Índices Selecionados e Criar o Subset ---
# Concatena os índices selecionados de ambas as classes
combined_subset_indices = np.concatenate([selected_benign_indices, selected_malicious_indices])

# Embaralha os índices combinados para misturar as classes no subset
np.random.shuffle(combined_subset_indices)

# Usa os índices combinados para criar os subsets de X_processed e labels
X_subset = X_processed[combined_subset_indices]
labels_subset = labels[combined_subset_indices]

# --- 6. Verificar o Subset Criado ---
print(f"\nShape do X_subset criado: {X_subset.shape}")
print(f"Shape do labels_subset criado: {labels_subset.shape}")
print(f"Contagem de classes no subset: Benigno={np.sum(labels_subset == 0)}, Malicioso={np.sum(labels_subset == 1)}")

states = [0,100,1000]
c = 0
for state in states:   
    for image_type, transformer in image_reshapes.items():
        features_np = []
        batch_size = 128
        for i in range(0,len(X_subset), batch_size):
            batch_domains = X_subset[i:i + batch_size]
            batch_features_gpu = transformer.fit_transform(batch_domains)
            features_np.append(batch_features_gpu)
            gc.collect() 
            print(f"Processado lote {i // batch_size + 1}/{(len(domain_urls) + batch_size - 1) // batch_size}")
        X_1d_transformed = np.concatenate(features_np,axis=0)
        print(f"Data after GAF processing: {len(X_1d_transformed)} Samples - {X_1d_transformed[0].shape} - Shape")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_1d_transformed, labels_subset, test_size=0.3, stratify=labels_subset, random_state=state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=state
        )
        base_dir = f"../datasets/domain-TFIDV/{str(image_type)}{c}"
        train_dir, val_dir, test_dir = [
            os.path.join(base_dir, d) for d in ["train", "val", "test"]
        ]
        for subdir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(subdir, "benign"), exist_ok=True)
            os.makedirs(os.path.join(subdir, "malicious"), exist_ok=True)
            
        datasets = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }
        for dataset_name, (X_data, y_data) in datasets.items():
            save_images(dataset_name, X_data, y_data, image_type)
            print(f"Imagens {image_type}, geradas e salvas com sucesso!")
        gc.collect()
    c+=1
