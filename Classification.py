import numpy as np
import torch
from support.data_loader import get_dataloader
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data_loader(train_path, batch_size, sample_len, return_recording=False, labels_info=None):
    data_loader = get_dataloader(recording_path=train_path, sample_rate=16000,
                                  batch_size=int(batch_size), sample_len_sec=sample_len,
                                  return_recording=return_recording, shuffled=False, label_dict = labels_info)
    return data_loader

def prediction_wav2vec(data_dir):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained("models/wav2vec2-adapted")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()  # put in inference mode
    results = []
    labels = []
    recordings = []
    timestamps = []
    for batch, label, recording, start_time in data_dir:
        inputs = processor(batch.squeeze().numpy(),
                           sampling_rate=16000,
                           return_tensors="pt",
                           padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        result = hidden_states.mean(dim=1)
        print(result.shape)
        results.extend(result.cpu().detach().numpy())
        try:
            labels.extend(label.cpu().detach().numpy())
        except:
            labels.extend(np.array(label))
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())
        # features: Tensor with shape [num_frames, feature_dim]
    result = np.array(results)
    labels = np.array(labels)
    return result, labels, recordings, timestamps

def fit_linear_layer(samples, labels, output_size, save_path):
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim
    import torch.nn.functional as F
    print('SIZE: ', samples.shape)

    classifier = torch.nn.Linear(samples.shape[1], output_size)

    X_tensor = torch.from_numpy(samples).float()
    y_tensor = torch.from_numpy(labels).long()

    dataset = TensorDataset(X_tensor, y_tensor)

    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        min_lr=1e-6
    )

    num_epochs = 150
    prev_loss = torch.inf
    final_classifier = classifier

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            y_pred = classifier(X_batch)

            y_onehot = F.one_hot(
                y_batch,
                num_classes=output_size
            ).float()

            loss = criterion(y_pred, y_onehot)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        # Update scheduler based on epoch loss
        scheduler.step(epoch_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {epoch_loss:.4f}, "
            f"LR: {current_lr:.6f}"
        )

        if epoch_loss < prev_loss:
            prev_loss = epoch_loss
            torch.save(classifier.state_dict(), save_path)

            final_classifier = classifier

    final_classifier.eval()
    return final_classifier

def compute_mAP(test_logits, labels):
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize
    proba = torch.sigmoid(test_logits)
    # proba = torch.softmax(test_logits, dim=1)
    labels_bin = label_binarize(labels, classes=np.arange(proba.shape[1]))
    mAP = average_precision_score(labels_bin, proba, average='macro')
    print('mAP: ', mAP)
    return mAP

def compute_ROC_multilabel_multiclass(test_logits, labels, output_size):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    proba = torch.sigmoid(test_logits)
    # proba = torch.softmax(test_logits, dim=1)
    labels_bin = label_binarize(labels, classes=np.arange(proba.shape[1]))
    # ROC-AUC for each class (one-vs-rest)
    auc_per_class = []

    for c in range(output_size):
        auc = roc_auc_score(
            labels_bin[:, c],
            proba[:, c]
        )
        auc_per_class.append(auc)

    # Macro ROC-AUC
    macro_auc = np.mean(auc_per_class)

    print("AUC per class:", auc_per_class)
    print("Macro ROC-AUC:", macro_auc)
    return macro_auc

def compute_weighted_ROC(test_logits,labels, output_size):
    from torchmetrics.classification import MultilabelAUROC
    import torch.nn.functional as F
    auroc = MultilabelAUROC(num_labels=output_size, average='weighted', thresholds=None)
    y_onehot = F.one_hot(
        torch.from_numpy(labels),
        num_classes=output_size
    ).int()

    score = auroc(test_logits, y_onehot)
    print("Weighted AUROC: ", score)
    return score

train_dir = '/projects/0/vusr0637/shipsEar/train/'
test_dir = '/projects/0/vusr0637/shipsEar/test/'

train_loader = load_data_loader(train_dir, 32, sample_len=5, return_recording=True)
test_loader = load_data_loader(test_dir, 32, sample_len=5, return_recording=True)

result_train, labels_train, recording_train, timestamps_train = prediction_wav2vec(train_loader)
result_test, labels_test, recording_test, timestamps_test = prediction_wav2vec(test_loader)

output_size = len(np.unique(labels_train))
print(np.unique(labels_train), np.unique(labels_test))
print('Output size: ', output_size)
reg = fit_linear_layer(result_train, labels_train, output_size, 'models/Classifiers/classifier_ShipsEar_MarineNet.pt')

X_test_tensor = torch.from_numpy(result_test).float()
X_train_tensor = torch.from_numpy(result_train).float()

with torch.no_grad():
    # X_test_tensor[:, 0] = 0
    logits_test = reg(X_test_tensor)
    test_predict = torch.argmax(logits_test, dim=1)
test_predict = test_predict.cpu().detach().numpy()

with torch.no_grad():
    logits_train = reg(X_train_tensor)
    train_predict = torch.argmax(logits_train, dim=1)
train_predict = train_predict.cpu().detach().numpy()

acc_score = accuracy_score(labels_test, test_predict)
print(test_predict)

print(acc_score)
print(confusion_matrix(labels_test, test_predict))

compute_mAP(test_logits=logits_test, labels=labels_test)
compute_ROC_multilabel_multiclass(logits_test, labels_test, output_size)
compute_weighted_ROC(logits_test, labels_test, output_size)