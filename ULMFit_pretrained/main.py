#system lib
import numpy as np 
import os
from sklearn.metrics import f1_score
#NLP
from torchtext.vocab import Vectors, GloVe
#torch
import torch
#custom
from dataloader import get_dataset
#from models.LSTM import LSTMClassifier
#from models.LSTM_ATTN import AttentionModel
#from models.RNN import RNN
#from models.RCNN import RCNN
#from models.RNN_ATTENTION import RNN_ATTENTION
#from models.RNN_mine import RNN_mine
from LSTM_SE import LSTM_SE
def train_model(model, train_iter, val_iter, optim, loss, num_epochs, batch_size=1):
   
    clip = 5
    val_loss_min = np.Inf
    
    total_train_epoch_loss = list()
    total_train_epoch_acc = list()
        
    total_val_epoch_loss = list()
    total_val_epoch_acc = list()
        
    best_epoch = 0
    best_f1 = 0
    best_acc = 0 
    for epoch in range(num_epochs):

        model.train()
        
        train_epoch_loss = list()
        train_epoch_acc = list()
        
        val_epoch_loss = list()
        val_epoch_acc = list()
        
        for idx, batch in enumerate(train_iter):
            
            text = batch.text[0]
            target = batch.target
            target = target - 1  # to make target begins from 0
            target = target.type(torch.LongTensor)

            text = text.to(device)
            target = target.to(device)
            
            optim.zero_grad()
            
            if text.size()[0] != batch_size:
                continue
            
            prediction = model(text)
            
            loss_train = loss(prediction.squeeze(), target)
            loss_train.backward()
            
            num_corrects = (torch.max(prediction, 1)[1].
                                view(target.size()).data == target.data).float().sum()
            acc = 100.0 * num_corrects / len(batch)
            train_epoch_loss.append(loss_train.item())
            train_epoch_acc.append(acc.item())
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            optim.step()
    
    
        model.eval()

        with torch.no_grad():
            for idx, batch in enumerate(val_iter):
                
                text = batch.text[0]
                target = batch.target
                target = target - 1
                target = target.type(torch.LongTensor)
                
                text = text.to(device)
                target = target.to(device)
                
                if text.size()[0] != batch_size:
                    continue
                
                logits = model(text)
                loss_val = loss(logits.squeeze(), target)
                predictions = torch.max(logits, 1)[1].view(target.size()).data
                f1 = f1_score(target.data.cpu(), predictions.cpu(), average='macro')
                num_corrects = (predictions == target.data).float().sum()
                acc = 100.0 * num_corrects / len(batch)

                val_epoch_loss.append(loss_val.item())
                val_epoch_acc.append(acc.item())
                
             
            if np.mean(val_epoch_acc) >= best_acc:
                best_f1 = f1
                best_acc = np.mean(val_epoch_acc)
                best_epoch = epoch
                torch.save(model.state_dict(), 'lstm_se_tweets.pth')
                val_loss_min = np.mean(val_epoch_loss)
            
            print('Epoch:%d, Current Validation Acc:%.4f Best_epoch:%d Best_acc:%.4f, Best_F1:%0.4f'
                    %(epoch,np.mean(val_epoch_acc),best_epoch, best_acc, best_f1))
    
    
            

        total_train_epoch_loss.append(np.mean(train_epoch_loss))
        total_train_epoch_acc.append(np.mean(train_epoch_acc))
    
        total_val_epoch_loss.append(np.mean(val_epoch_loss))
        total_val_epoch_acc.append(np.mean(val_epoch_acc))
    
    return (total_train_epoch_loss, total_train_epoch_acc,
            total_val_epoch_loss, total_val_epoch_acc)


def seed_everything(seed=27):
  #random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def update_with_pretrained_weights(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    diff = {k: v for k, v in model_dict.items() if \
            k in pretrained_dict and pretrained_dict[k].size() != v.size()}

    pretrained_dict.update(diff)
    model.load_state_dict(pretrained_dict)
    return model


if __name__ == "__main__":
    seed_everything()
    lr = 1e-4
    batch_size = 128
    output_size = 2
    hidden_size = 128
    embedding_length = 300
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vocab_size, word_embeddings, train_iter, val_iter = get_dataset(fix_length=25, train_dir = 'train.csv', 
            batch_size=batch_size, device=device)

    model = LSTM_SE(vocab_size=vocab_size, 
                        output_size=output_size, 
                        embedding_length=embedding_length,
                        hidden_size=hidden_size,batch_size=batch_size,
                        weights=word_embeddings
    )
    
    #model = update_with_pretrained_weights(model,"ULMFiT/sentiment_analysis_lstm_se.pth")
      
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
        
    train_loss, train_acc, val_loss, val_acc = train_model(model=model,train_iter=train_iter,val_iter=val_iter,
            optim=optim,loss=loss,num_epochs=100,batch_size=batch_size)


