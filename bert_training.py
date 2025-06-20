import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import evaluate
from transformers import DataCollatorWithPadding
from datasets import DatasetDict, Dataset

def preprocess_function(names):
    return tokenizer(names['name'], truncation=True)

accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    probabilities = np.exp(predictions) /np.exp(predictions).sum(-1, 
                                                                 keepdims=True)
    positive_class_probs = probabilities[:,1]
    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs,
                                     references=labels)['roc_auc'],4)
    
    predicted_classes = np.argmax(predictions, axis=1)
    acc = np.round(accuracy.compute(predictions=predicted_classes, 
                                    references=labels)['accuracy'],4)
    return {"Accuracy": acc, "AUC": auc}

df = pd.read_csv("dataset.csv")

labels = df['malicious'].values
names = df['name'].values
df = None

print(f"First 5 domain: {names[:5]}")
print(f"First 5 labels: {labels[:5]}")

X_train, X_temp, y_train, y_temp = train_test_split(
    names, labels, test_size=0.30, random_state=0, stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
)

print("\n--- Tamanho dos Conjuntos ---")
print(f"Treino: {len(X_train)} amostras ({len(X_train)/len(names):.2%})")
print(f"Validação: {len(X_val)} amostras ({len(X_val)/len(names):.2%})")
print(f"Teste: {len(X_test)} amostras ({len(X_test)/len(names):.2%})")
print("---------------------------\n")

train_dataset = Dataset.from_dict({'name': X_train, 'label': y_train})
val_dataset = Dataset.from_dict({'name': X_val, 'label': y_val})
test_dataset = Dataset.from_dict({'name': X_test, 'label': y_test})


raw_datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})



model_path = "bert-case-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)

id2label  = {0:"Benign", 1:"Malicious"}
label2id = {"Benign":0, "Malicious":1}

model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                           num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id,)

for name, param in model.base_model.named_parameters():
    param.requires_grad = False

for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

tokenized_names = raw_datasets.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

lr = 2e-4
batch_size = 128
num_epochs = 20

training_args = TrainingArguments(
    output_dir="bert-domain-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="AUC",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_names["train"],
    eval_dataset=tokenized_names["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions = trainer.predict(tokenized_names["test"])

logits = predictions.predictions
labels = predictions.labels_ids


print("Avaliação no Conjunto Final de teste dos domínios!")
metrics = compute_metrics((logits,labels))
print(metrics)
