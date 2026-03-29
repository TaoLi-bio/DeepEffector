import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from Funmodel import Model
import numpy as np
import torch
import torch.nn as nn


class CustomDataset(Dataset):
    def __init__(self, features1, labels):
        self.features1 = torch.tensor(features1.values.astype(np.float32))
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features1)

    def __getitem__(self, index):
        x1 = self.features1[index]
        y = self.labels[index]
        return x1, y


train = pd.read_excel("data")
x_train = train.iloc[:, :-1]
x_train_label = train.iloc[:, -1]

num_epochs = 100
batch_size = 32
num_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


metrics_list = []


kf = KFold(n_splits=num_folds, shuffle=True, random_state=20)
tprs, fprs, precisions, recalls = [], [], [], []

for fold, (train_index, val_index) in enumerate(kf.split(train)):
    train_ProtT5 = x_train.iloc[train_index]
    train_labels = x_train_label.iloc[train_index]
    val_ProtT5 = x_train.iloc[val_index]
    val_labels = x_train_label.iloc[val_index]

    train_dataset = CustomDataset(train_ProtT5, train_labels)
    val_dataset = CustomDataset(val_ProtT5, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    cnn_model = Model()
    cnn_model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.00001)

    cnn_model.train()
    for epoch in range(num_epochs):
        for data2, labels in train_loader:
            data2 = data2.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            final_output = cnn_model(data2.unsqueeze(1))
            loss = criterion(final_output[:, 0], labels)
            loss.backward()
            optimizer.step()

    all_predictions = []
    all_labels = []
    all_auc = []
    for data2, labels in val_loader:
        data2 = data2.to(device)
        final_output = cnn_model(data2.unsqueeze(1))
        scores = final_output[:, 0].tolist()
        all_auc.extend(scores)
        final_output = (final_output.data > 0.5).int()
        all_labels.extend(labels.tolist())
        all_predictions.extend(final_output.tolist())

    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_precision = precision_score(all_labels, all_predictions)
    val_roc = roc_auc_score(all_labels, all_auc)
    val_recall = recall_score(all_labels, all_predictions)
    val_f1 = f1_score(all_labels, all_predictions)
    precision, recall, _ = precision_recall_curve(all_labels, all_auc)
    fpr, tpr, _ = roc_curve(all_labels, all_auc)
    val_pr_auc = auc(recall, precision)


    metrics_list.append({
        'Accuracy': val_accuracy,
        'Precision': val_precision,
        'Recall': val_recall,
        'F1': val_f1,
        'ROC_AUC': val_roc,
        'PR_AUC': val_pr_auc
    })


    num_samples = 100
    precision_sampled = np.linspace(0, 1, num_samples)
    recall_sampled = np.interp(precision_sampled, precision, recall)
    fpr_sampled = np.linspace(0, 1, num_samples)
    tpr_sampled = np.interp(fpr_sampled, fpr, tpr)

    fprs.append(fpr_sampled)
    tprs.append(tpr_sampled)
    precisions.append(precision_sampled)
    recalls.append(recall_sampled)

    print(f"Fold {fold + 1} - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, ROC_AUC: {val_roc:.4f}, PR_AUC: {val_pr_auc:.4f}")


mean_metrics = {metric: np.mean([fold_metrics[metric] for fold_metrics in metrics_list]) for metric in metrics_list[0]}
print("\nAverage metrics across 5 folds:")
for metric, value in mean_metrics.items():
    print(f"{metric}: {value:.4f}")


mean_precision = np.mean(precisions, axis=0)
mean_recall = np.mean(recalls, axis=0)
mean_fpr = np.mean(fprs, axis=0)
mean_tpr = np.mean(tprs, axis=0)


pr_curve_data = pd.DataFrame({'Precision': mean_precision, 'Recall': mean_recall})
pr_curve_data.to_csv('PRclassifier_5cv.csv', index=False)

roc_curve_data = pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr})
roc_curve_data.to_csv('ROCclassifier_5cv.csv', index=False)


final_dataset = CustomDataset(x_train, x_train_label)
final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True)

final_model = Model()
final_model.to(device)
final_optimizer = optim.Adam(final_model.parameters(), lr=0.00001)
final_model.train()

for epoch in range(num_epochs):
    for data2, labels in final_loader:
        data2 = data2.to(device)
        labels = labels.to(device)
        final_optimizer.zero_grad()
        final_output = final_model(data2.unsqueeze(1))
        loss = criterion(final_output[:, 0], labels)
        loss.backward()
        final_optimizer.step()


torch.save(final_model.state_dict(), "final_classifier.pt")
print("Final model trained on all data and saved.")
