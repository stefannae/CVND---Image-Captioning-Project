import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(pretrained=True)  # change it for other pretrained models
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(resnet.fc.in_features, embed_size)


    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)

        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, batch_size=10, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        

    def forward(self, features, captions):
        features = features.unsqueeze(1)
        embeds = self.embed(captions)
        inputs = (features, embeds)
        inputs = torch.cat(inputs, 1)
        
        outputs, _ = self.lstm(inputs)
        outputs = self.linear(outputs)
        
        return outputs[:,:-1,:]


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence_ids = []
        for _ in range(max_len):
            output, states = self.lstm(inputs, states)
            output = self.linear(output)
            _, topi = output.topk(1)
            sentence_ids.append(topi)
            inputs = self.embed(topi)
            inputs = torch.squeeze(inputs, 0)

        return [i.item() for i in sentence_ids]
