# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from data.puf_dataset import PufDataset
from concern.config import Configurable, Config
import argparse
import cv2
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description="Test")
args = parser.parse_args()

conf = Config()
Trainsetting_conf=conf.compile(conf.load('./configs/seg_base.yaml'))
Trainsetting_conf.update(cmd=args)
cmd_in = vars(Trainsetting_conf.pop('cmd', dict()))
cmd_in.update(is_train=True)

main_cfg=Trainsetting_conf['Experiment']['main']
train_tag=main_cfg['train']
valid_tag=main_cfg['valid']
train_synth_cfg=Trainsetting_conf['Experiment'][train_tag]['data_loader']['dataset']
valid_synth_cfg=Trainsetting_conf['Experiment'][valid_tag]['data_loader']['dataset']

train_synth_dataset=Configurable.construct_class_from_config(train_synth_cfg)
valid_synth_dataset=Configurable.construct_class_from_config(valid_synth_cfg)
#train_synth_cfg.update(cmd=cmd_in)
#train_synth_img_loader = Configurable.construct_class_from_config(train_synth_cfg)
img, clsTpe =train_synth_dataset[1]
tmp=train_synth_dataset.processes[1].lib_inv_trans(img)

bs = 100

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])
#    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

train_loader = torch.utils.data.DataLoader(dataset=train_synth_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=valid_synth_dataset, batch_size=bs, shuffle=False)





class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))






# build network
z_dim = 100
#mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)
mnist_dim = img.shape[1] *  img.shape[2]

G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)



# loss
criterion = nn.BCELoss()

# optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)




def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(x.shape[0], 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(x.shape[0], z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(x.shape[0], 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()




def G_train(x):

    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()


load_model=False
if(load_model):
    G.load_state_dict(torch.load('./model_save_G.pth'))
    D.load_state_dict(torch.load('./model_save_D.pth'))


n_epoch = 50
for epoch in range(1, n_epoch+1):           
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        #print('batch_idx', batch_idx)
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))


torch.save(G.state_dict(), './model_save_G.pth')
torch.save(D.state_dict(), './model_save_D.pth')


with torch.no_grad():
    test_z = Variable(torch.randn(bs, z_dim).to(device))
    generated = G(test_z)
    print("Generated size = ", generated.size)

    save_image(generated.view(generated.size(0), 1, img.shape[1], img.shape[2]), './samples/sample_' + '.png')
