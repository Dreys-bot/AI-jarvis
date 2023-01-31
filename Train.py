import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from NeuralNetwork import bag_of_words, tokenize, stem
from Brain import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] #Dictonnary(pattern, tag)

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

#result: [(['hello'], 'greeting'), (['hii'], 'greeting'), (['hey'], 'greeting'), (['wake', 'up'], 'greeting'), (['jarvis'], 'greeting'), (['bye'], 'bye'), (['good', 'bye'], 'bye'), (['see', 'you', 'later'], 'bye'), (['sleep'], 'bye'), (['exit'], 'bye'), (['how', 'are', 'you'], 'health'), (['how', 'you'], 'health'), (['are', 'you', 'fine', '?'], 'health'), (['are', 'you', 'good'], 'health')]
ignore_words = [',', '?', '/', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words] #For each pattern, add a radical of it in all_words array

all_words = sorted(set(all_words))
tags = sorted(set(tags))


#Here, we have a array (all_words) which contains the radical of all patterns and a array (xy) which contains a dictionnary like (pattern_tokenized, tag)
#all_words = ['hello', 'hii', 'hey', 'wake', 'up', 'jarvi', 'bye', 'good', 'bye', 'see', 'you', 'later', 'sleep', 'exit', 'how', 'are', 'you', 'how', 'you', 'are', 'you', 'fine', 'are', 'you', 'good']
#tags = ['bye', 'greeting', 'health']
#xy = [(['hello'], 'greeting'), (['hii'], 'greeting'), (['hey'], 'greeting'), (['wake', 'up'], 'greeting'), (['jarvis'], 'greeting'), (['bye'], 'bye'), (['good', 'bye'], 'bye'), (['see', 'you', 'later'], 'bye'), (['sleep'], 'bye'), (['exit'], 'bye'), (['how', 'are', 'you'], 'health'), (['how', 'you'], 'health'), (['are', 'you', 'fine', '?'], 'health'), (['are', 'you', 'good'], 'health')]

#Continue...

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)


    label = tags.index(tag)
    y_train.append(label)
  

    
x_train = np.array(x_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)


#Our x Train dataset is a numpy array. if 1, it means that it the position of the pattern
print("Training The Model..")

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()

#dataloader is for avoid overfitting. DIvide data in many batchs
train_loader = DataLoader(dataset = dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = 0)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#######Training per epoch

#Hypothèse(par forcement vrai): The model is use to reconize with tags we are on by using a word in a sentence. L'utilisateur dit une phrase, l'ordinateur fait un numpy array qui represente la position des mots de sa phrase et predict un label(tag) en fonctin de ce qu'il a appris. Pas besoin de comprendre le sens de la phrase.
#(words, labels) represent (batch of x_train shuffled, labels of the batch) like (tensor([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                                                                                                #[0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                                                                                                #[1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                                                                                                                #[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                                                                                                #[1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
                                                                                                                                #[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                                                                                                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.],
                                                                                                                                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]]), tensor([1, 0, 2, 1, 2, 0, 0, 1], dtype=torch.int32))
       



for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') 

print(f'Final Loss: {loss.item():.4f}')   

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "TrainData.pth"
torch.save(data, FILE)

print(f'Training Complete, File Saved To {FILE}')