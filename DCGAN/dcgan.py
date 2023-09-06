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

bs = 200

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


def initialize_weights (model):
    for m in model.modules ():
        if isinstance (m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_ (m.weight.data, 0.0, 0.02)

class Discriminator_dc (nn.Module):
    """ a model that judges between real and fake images """
    def __init__ (self, img_channels, features_d):
        super (Discriminator_dc, self).__init__ ()
        
        # Input: N x channels_img x 128 x 128
        self.disc = nn.Sequential (
            nn.Conv2d (img_channels, features_d, kernel_size = 4, stride = 2, padding = 1), # 64x64
            nn.LeakyReLU (0.2),
            self._block (features_d * 1, features_d * 2, 4, 2, 1),                          # 32x32
            self._block (features_d * 2, features_d * 4, 4, 2, 1),                          # 16x16
            self._block (features_d * 4, features_d * 8, 4, 2, 1),                          # 8x8
            self._block (features_d * 8, features_d * 16, 4, 2, 1),                         # 4x4
            nn.Conv2d (features_d * 16, 1, kernel_size = 4, stride = 2, padding = 0),        # 1x1
            nn.Sigmoid ()
        )
      
    def _block (self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential (
            nn.Conv2d (in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d (out_channels),
            nn.LeakyReLU (0.2)
        )
    
    def forward (self, x):
        return self.disc (x)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-30T11:43:27.25327Z","iopub.execute_input":"2021-12-30T11:43:27.25348Z","iopub.status.idle":"2021-12-30T11:43:27.266917Z","shell.execute_reply.started":"2021-12-30T11:43:27.253432Z","shell.execute_reply":"2021-12-30T11:43:27.266199Z"}}
class Generator_dc (nn.Module):
    """ a model generates fake images """
    def __init__ (self, z_dim, img_channels, features_g):
        super (Generator_dc, self).__init__ ()
        
        # Input: N x z_dim x 1 x 1
        self.gen = nn.Sequential (
            self._block (z_dim, features_g * 32, 4, 2, 0),                               # 4x4
            self._block (features_g * 32, features_g * 16, 4, 2, 1),                      # 8x8
            self._block (features_g * 16, features_g * 8, 4, 2, 1),                       # 16x16
            self._block (features_g * 8, features_g * 4, 4, 2, 1),                       # 32x32
            self._block (features_g * 4, features_g * 2, 4, 2, 1),                       # 64x64
            nn.ConvTranspose2d (
                features_g * 2, img_channels, kernel_size = 4, stride = 2, padding = 1), # 128x128
            nn.Tanh ()
        )
        
        
        
    def _block (self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential (
            nn.ConvTranspose2d (in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d (out_channels),
            nn.ReLU ()
        )
    
    def forward (self, x):
        return self.gen (x)





class Discriminator_gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bnorm1 = nn.BatchNorm2d(8)
        
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv4 = nn.Conv2d(16, 32, 3)
        self.bnorm2 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(32, 4, 3)
        
        self.fc1 = nn.Linear(5776, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
    def forward(self, x):
        pred = F.leaky_relu(self.conv1(x.reshape(-1,1,48,48)))
        pred = F.leaky_relu(self.bnorm1(self.conv2(pred)))
        pred = F.leaky_relu(self.conv3(pred))
        pred = F.leaky_relu(self.bnorm2(self.conv4(pred)))     
        pred = F.leaky_relu(self.conv5(pred))
        
        pred = pred.reshape(-1, 5776)

        pred = F.leaky_relu(self.fc1(pred))
        pred = F.leaky_relu(self.fc2(pred))
        pred = torch.sigmoid(self.fc3(pred))
        
        return pred
    
class Generator_gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 5776)

        self.convT1 = nn.ConvTranspose2d(4, 32, 3)       
        self.convT2 = nn.ConvTranspose2d(32, 16, 3)
        self.bnorm1 = nn.BatchNorm2d(16)
        self.convT3 = nn.ConvTranspose2d(16, 8, 3)
        self.convT4 = nn.ConvTranspose2d(8, 4, 3)
        self.bnorm2 = nn.BatchNorm2d(4)
        self.convT5 = nn.ConvTranspose2d(4, 1, 3)
        
    def forward(self, x):
        pred = F.leaky_relu(self.fc1(x))
        pred = F.leaky_relu(self.fc2(pred))
        pred = F.leaky_relu(self.fc3(pred))
        
        pred = pred.reshape(-1, 4, 38, 38)
        
        pred = F.leaky_relu(self.convT1(pred))
        pred = F.leaky_relu(self.bnorm1(self.convT2(pred)))
        pred = F.leaky_relu(self.convT3(pred))
        pred = F.leaky_relu(self.bnorm2(self.convT4(pred)))
        pred = torch.sigmoid(self.convT5(pred))
        
        return pred



class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
        #output : 17280
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 4096)
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

#G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
#D = Discriminator(mnist_dim).to(device)

D = Discriminator_dc (img_channels = 1, features_d = 16).to(device)
G = Generator_dc (z_dim = 100, img_channels = 1, features_g = 160).to(device)

initialize_weights (D)
initialize_weights (G)


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
    #x_real, y_real = x.view(-1, mnist_dim), torch.ones(x.shape[0], 1)
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output.view(-1,1), y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(x.shape[0], z_dim,1, 1).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(x.shape[0], 1).to(device))

    D_output = D(x_fake) 
    D_output = D_output.reshape(-1,1)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = (D_real_loss + D_fake_loss)/2
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()




def G_train(x):

    #=======================Train the generator=======================#
    G.zero_grad()

    #z = Variable(torch.randn(bs, z_dim).to(device))
    z = Variable(torch.randn(bs, z_dim, 1, 1).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output).reshape(-1, 1)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()


load_model=False
if(load_model):
    G.load_state_dict(torch.load('./model_save_G.pth'))
    D.load_state_dict(torch.load('./model_save_D.pth'))


n_epoch = 100
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
    test_z = Variable(torch.randn(bs, z_dim, 1, 1).to(device))
    generated = G(test_z)
    print("Generated size = ", generated.size)

    save_image(generated.view(generated.size(0), 1, img.shape[1], img.shape[2]), './samples/sample_' + '.png')

    generated_inv =train_synth_dataset.processes[1].inv_normal(generated)
    save_image(generated_inv.view(generated_inv.size(0), 1, img.shape[1], img.shape[2]), './samples/sample_inv' + '.png')
