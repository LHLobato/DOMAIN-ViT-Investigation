import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np 
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader 
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import time
import os
import torch.backends.cudnn as cudnn
import config
from transformers import get_cosine_schedule_with_warmup
from transformers import ViTForImageClassification, ViTConfig
#Laço de repetição que permite fazer os testes com as três divisões de dados que foram definidas

def find_optimal_threshold(labels, preds_proba):
    """Encontra o limiar que maximiza a acurácia."""
    thresholds = np.linspace(0.3, 0.7, 100)  # Foca na região crítica (evita extremos)
    accuracies = [accuracy_score(labels, preds_proba >= t) for t in thresholds]
    return thresholds[np.argmax(accuracies)]



def apply_optimal_threshold(labels, preds_proba, threshold):
    """Aplica o limiar ótimo e retorna métricas."""
    preds_proba = np.array(preds_proba).flatten()
    preds_class = (preds_proba >= threshold).astype(int)
    acc = accuracy_score(labels, preds_class)
    prec = precision_score(labels, preds_class, zero_division=0)
    recall = recall_score(labels, preds_class, zero_division=0)
    f1 = f1_score(labels, preds_class)
    auc = roc_auc_score(labels, preds_proba) if len(np.unique(labels)) > 1 else None
    return acc, prec, recall, f1, auc
    #---- Funcao que salvara o modelo de acordo com o diretorio passado
    
def save_model(model, path):
        torch.save(model.state_dict(),path)

    #---- Função de treinamento e validação do modelo
    
def train(model, num_epochs, train_loader, val_loader, paticence = 10): 
        start_time = time.time()
        scaler = torch.amp.GradScaler()
        print(f'Iniciando treinamento...')
        epoch_counter = 0
        paticence_limit = 0
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.vit.parameters() if not any(p is cp for cp in model.classifier.parameters())],'lr':3e-4, 'weight_decay':0.05},
            {'params': model.classifier.parameters(), 'lr': 1e-4},     
            ])

        scheduler = scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10 * len(train_loader),  # Warmup de 5 épocas
            num_training_steps=100 * len(train_loader),  # Total de épocas
        )

        best_epoch = 0
        best_accuracy = float('-inf')
        current_lr = 0
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            all_preds, all_labels = [], []
            print(f'Epoch: {epoch + 1}....')

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    
                    loss = criterion(outputs.logits, labels)
                scaler.scale(loss).backward()  # Substitui loss.backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()

                all_preds.extend(outputs.logits.detach().cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())
            
            accuracy, precision, recall, f1, auc = apply_optimal_threshold(all_labels, all_preds, 0.5)
                        
            #---- Etapa de validacao
            
            model.eval()
            val_loss = 0.0
            val_preds_proba, va_labels = [], []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs.logits, labels)
                    val_loss += loss.item()

                    val_preds_proba.extend(outputs.logits.detach().cpu().numpy().flatten())
                    va_labels.extend(labels.cpu().numpy())
                    
            optimal_threshold = find_optimal_threshold(va_labels, val_preds_proba)
            val_accuracy, val_precision, val_recall, val_f1, val_auc = apply_optimal_threshold(
                va_labels, val_preds_proba, optimal_threshold
            )
            scheduler.step()

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch + 1
                paticence_limit  = 0
                best_threshold = optimal_threshold  # Salva o limiar ótimo
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimal_threshold': best_threshold,
                }, save_dir + save_model_name + f"{num_epochs}.pth")
            else:
                paticence_limit +=1
            # Logs de treinamento
            epoch_counter+=1
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"| Train Loss: {running_loss / len(train_loader):.4f} | "
                f"Train Acc: {accuracy:.4f} | Train Prec: {precision:.4f} | Train Rec: {recall:.4f} | Train F1: {f1:.4f} | Train ROC-AUC: {auc:.4f}")
            print(f"Val Loss: {val_loss / len(val_loader):.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Val Prec: {val_precision:.4f} | "
                f"Val Rec: {val_recall:.4f} | "
                f"Val F1: {val_f1:.4f}|"
                f"Val ROC-AUC: {val_auc:.4f}")
            
            if current_lr != optimizer.param_groups[0]['lr']:
                for i, group in enumerate(optimizer.param_groups):
                    print(f"Grupo {i}: LR = {group['lr']}, Parâmetros = {len(group['params'])}")
                current_lr = optimizer.param_groups[0]['lr']
            
            if paticence_limit >=paticence:
                print(f"Early Stopping triggered with {epoch_counter} epochs!")
                break
        total_time = (time.time() - start_time) / 60
        
        print(f'Training Took: {total_time:.2f} minutes!')
        print(f'With an average of {total_time /epoch_counter:.2f} minutes per epoch!')
        
        return best_accuracy, best_epoch

    #----Função de Teste do modelo.
def test(model, test_loader,optimal_threshold):
        test_preds = []
        test_labels = []
        test_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        model.eval()  

        with torch.no_grad():  
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs.logits, labels)
                test_loss += loss.item()

                test_preds.extend(outputs.logits.detach().cpu().numpy().flatten())
                test_labels.extend(labels.cpu().numpy())

        test_accuracy, test_precision, test_recall, test_f1, test_auc = apply_optimal_threshold(test_labels, test_preds, optimal_threshold)
        # Logs de teste
        print(f"Test Loss: {test_loss / len(test_loader):.4f} | "
            f"Test Acc: {test_accuracy:.4f} | "
            f"Test Prec: {test_precision:.4f} | "
            f"Test Rec: {test_recall:.4f} | "
            f"Test F1: {test_f1:.4f}|"
            f"Test ROC-AUC: {test_auc:.4f}")

        return test_loss / len(test_loader), test_accuracy, test_precision, test_recall, test_f1, test_auc

save_dir = "../modelos/VIT/"
test_dataset_dir = "../results/test0_https.csv"
image_names = ["GASF", "GADF", "RPLOT"]
ig_count=0
for data_dir in config.DATA_DIRS:    
    for i in range(config.NUM_DATASETS):
        #---- Diretorios que serao utilizados seja para carregar as imagens.
        #----Diretório de salvamento do modelo, e do resultado dos testes.
        #----Transform que carrega as imagens para as dimensões corretas. 
        transform = transforms.Compose([
            transforms.Resize((config.RESOLUTION,config.RESOLUTION )),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224, 0.225])
            #removendo normalização do ImageNet que estava sendo aplicada. 
        ])
        
        #-----Definição dos datasets.
        train_dataset = datasets.ImageFolder(root = data_dir[i]["train"], transform=transform)
        val_dataset = datasets.ImageFolder(root = data_dir[i]["val"], transform=transform)
        test_dataset = datasets.ImageFolder(root = data_dir[i]["test"], transform=transform)
        
        #-----Definição dos DataLoaders.
        test_loader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE, shuffle=False,num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size = config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, shuffle=True,num_workers=8, pin_memory=True)

        #---- Definição do Nome do modelo que será salvo.
        #-----Função que calcula as métricas de acordo com as etapas de examinação.
        
        print("Dados carregados com sucesso...")
        for num_layer in config.UNFROZEN_LAYERS:
            epochs = [config.NUM_EPOCHS]
            save_model_name = f"ViT{i}{image_names[ig_count]}Domain{num_layer}unfrozen{config.current_dataset}"
            for num_epochs in epochs:
                model = ViTForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224-in21k",  # Modelo pré-treinado para 224x224
                    num_labels=1,  # Classificação binária
                    ignore_mismatched_sizes=True,  # Permite substituir o head de classificação
                )
                model.classifier = nn.Linear(model.config.hidden_size, 1)
                for params in model.parameters():
                    params.requires_grad = False
                    
                for layer in model.vit.encoder.layer[-num_layer:]:  # Descongela as últimas 4 camadas
                    for param in layer.parameters():
                        param.requires_grad = True

                for params in model.classifier.parameters():
                    params.requires_grad = True
                    
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cudnn.benchmark = True
                
                model = model.to(device)

                best_train_acc, best_epoch= train(model, num_epochs, train_loader, val_loader)

                checkpoint = torch.load(save_dir + save_model_name + f"{num_epochs}.pth")
                model.load_state_dict(checkpoint['model_state_dict'])
                optimal_threshold = checkpoint['optimal_threshold']


                loss, acc, prec, rec, f1, auc = test(model, test_loader, optimal_threshold)

                print(f'Melhor acurácia de treinamento: {best_train_acc} atingida com {best_epoch} épocas')

            #---- Chamada das funcoes.
                data = {"Image_Dataset": f"{image_names[ig_count]}{i}-{config.current_dataset}",
                    "Model": "ViT", 
                    "Epochs": config.NUM_EPOCHS,
                    "Test_Loss":loss,
                    "Test_Acuracia": acc, 
                    "Test_Precisao": prec, 
                    "Test_Recall": rec, 
                    "Test_F1-Score": f1,
                    "Test_ROC-AUC": auc,
                    "Best_Acc_Train":best_train_acc,
                    "Best_Epoch_Acc":best_epoch,
                    "Num_Samples":config.NUM_SAMPLES,
                    "Dropout":config.P,
                    "Resolution":config.RESOLUTION,
                    "Unfrozen_Layers":num_layer+1,
                    "Data Normalization": 'Yes', 
                    "Weight Decay": config.WEIGHT_DECAY,
                    "WarmuP": 'Yes'
                    }

                test_df = pd.DataFrame([data])

                if os.path.exists(test_dataset_dir):
                    test_df.to_csv(test_dataset_dir, mode='a', header=False, index=False)
                else:
                    test_df.to_csv(test_dataset_dir, mode='w', header=True, index=False)
    ig_count+=1
            #---- Etapa que salva os dados obtidos a partir dos tetes, sinalizando o estado do modelo e tecnicas utilazadas para obter tais metricas.




