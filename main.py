from torchvision import datasets
from torch.utils.data import DataLoader,Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder='D:\\SOFTWARE PROJECT\\neutral network\\Pytorch\\5.fanshion_MINIST\\data'
fminist=datasets.FashionMNIST(data_folder,train=True,download=True)
tr_images=fminist.data
tr_labels=fminist.targets

val_fminist=datasets.FashionMNIST(data_folder,train=False,download=True)
val_images=val_fminist.data
val_labels=val_fminist.targets
unique_labels=tr_labels.unique()
class FashionMNISTDataset(Dataset):
    def __init__(self,x,y):
        x=x.float()/255#缩放数据集在0-1之间,加快loss和accuracy的收敛速度
        x=x.view(-1,28*28)
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        x=self.x[idx]
        y=self.y[idx]
        return x.to(device),y.to(device)
def data_loader():
    dataset=FashionMNISTDataset(tr_images,tr_labels)
    dataloader=DataLoader(dataset,batch_size=32,shuffle=True)
    va_dataset=FashionMNISTDataset(val_images,val_labels)
    val_dataloader=DataLoader(va_dataset,batch_size=len(val_images),shuffle=False)
    return dataloader,val_dataloader

def get_model():
    model=nn.Sequential(
        nn.Linear(28*28,1000),
        nn.ReLU(),
        nn.Linear(1000,10)
    ).to(device)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=1e-2)
    return model,loss_fn,optimizer
def train_batch(x,y,model,loss_fn,optimizer):
    model.train()
    optimizer.zero_grad()
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    loss.backward()
    optimizer.step()
    return loss.item()
@torch.no_grad()
def evaluate_batch(x,y,model,loss_fn):
    model.eval()
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    max_values,argmax=y_pred.max(-1)
    is_correct=argmax==y #is_correct是每一轮预测是否正确的布尔值列表，一共10个
    return is_correct.cpu().numpy().tolist()
@torch.no_grad()
def val_loss(x,y,model,loss_fn):#validation loss和trainingbatch的区别在于没有回测loss.backward()和optimizer.step()
    model.eval()
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    return loss.item()
trn_dl,val_dl=data_loader()
model,loss_fn,optimizer=get_model()
losses,accuracies=[],[]
val_losses,val_accuracies=[],[]
n_epochs=10
for epoch in range(n_epochs):
    epochs_losses,epochs_accuracies=[],[]
    val_epoch_losses,val_epoch_accuracies=[],[]
    for ix,batch in enumerate(iter(trn_dl)):
        x,y=batch
        loss=train_batch(x,y,model,loss_fn,optimizer)
        epochs_losses.append(loss)
    epoch_loss=np.array(epochs_losses).mean()
    losses.append(epoch_loss)
    for ix,batch in enumerate(iter(trn_dl)):
        x,y=batch
        is_correct=evaluate_batch(x,y,model,loss_fn)
        epochs_accuracies.extend(is_correct)
    epoch_accuracy=np.mean(epochs_accuracies)
    accuracies.append(epoch_accuracy)
    for ix,batch in enumerate(iter(val_dl)):
        x,y=batch
        val_is_correct=evaluate_batch(x,y,model,loss_fn)
        val_epoch_loss=val_loss(x,y,model,loss_fn)
        val_epoch_losses.append(val_epoch_loss)
        val_epoch_accuracies.extend(val_is_correct)
    val_epoch_accuracy=np.mean(val_epoch_accuracies)
    val_accuracies.append(val_epoch_accuracy)
    val_epoch_loss=np.mean(val_epoch_losses)
    val_losses.append(val_epoch_loss)  
epochs=np.arange(1,n_epochs+1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title('Loss vs Epochs')
plt.plot(epochs,losses,'-bo',label='training loss')
plt.plot(epochs,val_losses,'-ro',label='validation loss')
plt.legend(loc='upper left')
plt.subplot(1,2,2)
plt.title('Accuracy vs Epochs')
plt.plot(epochs,accuracies,'-bo',label='training accuracy')
plt.plot(epochs,val_accuracies,'-ro',label='validation accuracy')
plt.legend()
'''
print(f'tr_images & tr_labels shape: \n\tX-{tr_images.shape},\n\tY-{tr_labels.shape}')
print(f'\n\tZ-Unique labels in tr_labels: {unique_labels}')
print(f'\n\tNumber of unique labels in tr_labels: {len(unique_labels)}')
print(f'\n\tUnique classes:{fminist.classes}')
'''
R,C=len(unique_labels),10
fig,ax=plt.subplots(R,C,figsize=(C,R))
for label_class,plt_row in enumerate(ax):
    label_x_rows=torch.where(tr_labels==label_class)[0]
    for plt_ceil in plt_row:
        rand_index=torch.randint(0,len(label_x_rows),(1,)).item()
        label_index=label_x_rows[rand_index]
        img=tr_images[label_index]
        plt_ceil.imshow(img,cmap='gray')
        plt_ceil.axis('off')
plt.show()
