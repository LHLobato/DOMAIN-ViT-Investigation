from numpy import sqrt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField, RecurrencePlot
import matplotlib.pyplot as plt
import os

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

def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result

def extract_features(text, model, tokenizer, device='cuda'):
    model = model.to(device)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Todas as camadas
    
    # Combinação vetorizada das últimas 4 camadas
    last_four_layers = torch.cat(hidden_states[-4:], dim=-1)
    
    # Pooling com máscara de atenção (ignora padding)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_four_layers.size()).float()
    sum_embeddings = torch.sum(last_four_layers * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    features = sum_embeddings / sum_mask
    
    return features

bad_requests = loadData('PreProcessedAnomalous.txt')
good_requests = loadData('PreprocessedNormalTraining.txt')

all_requests = bad_requests + good_requests
labels_Bad = [1] * len(bad_requests)
labels_Good = [0] * len(good_requests)
labels = labels_Bad + labels_Good

print("All request loaded")
print("All labels signed")

model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
print("Model Loaded")

features = []
for i in range(len(all_requests)):
    features.append(extract_features(all_requests[i],model, tokenizer))
features = torch.cat(features).cpu().numpy()
print("Shape of Samples after Feature Extraction", features.shape)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=0
)


clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
test_auc = roc_auc_score(y_test, y_probs)
print("-------FOR BERT TOKENIZER-----------")
print("AUC no conjunto de teste:", test_auc)
print("Acurácia: ", accuracy_score(y_test, y_pred))
print(f"Desempenho atingido com resolução: {sqrt(len(features))}")

pipeline = Pipeline([
    ('pca', PCA(n_components=64)),
    ('scaler', MinMaxScaler())
])
print("Shape of Samples after Feature Extraction", features.shape)
features = pipeline.fit_transform(features)
print(f"Shape of Unique Feature after resize (image): {features[0].shape}")

image_reshapes = {
    "GASF": GramianAngularField(method = "summation"),
    "GADF": GramianAngularField(method = "difference"),
    "RPLOT": RecurrencePlot(dimension=1,threshold='point', percentage=20) 
}
i=0
states = [0,100,1000]
for state in states:   
    for image_type, transformer in image_reshapes.items():
        X_1d_transformed = transformer.fit_transform(features)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_1d_transformed, labels, test_size=0.3, stratify=labels, random_state=state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=state
        )
        base_dir = f"datasets/BERT-PREPROCESSED/{str(image_type)}+{i}"
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
    i+=1
