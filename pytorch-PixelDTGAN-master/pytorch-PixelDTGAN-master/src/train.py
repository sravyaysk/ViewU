# speed up the loading of the training data
import cv2
import numpy as np
import torch as th
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.model import NetG, NetD, NetA
from src.data_set import LookbookDataset
import torch.optim as optim
import visdom
from torchvision.utils import make_grid

vis = visdom.Visdom(port=5274)
win = None
win1 = None
netg = NetG()
netd = NetD()
neta = NetA()
netg.train()
netd.train()
neta.train()
device = th.device("cpu")

# weights init
all_mods = itertools.chain()
all_mods = itertools.chain(all_mods, [
    list(netg.children())[0].children(),
    list(netd.children())[0].children(),
    list(neta.children())[0].children()
])
for mod in all_mods:
    if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.ConvTranspose2d):
        init.normal_(mod.weight, 0.0, 0.02)
    elif isinstance(mod, nn.BatchNorm2d):
        init.normal_(mod.weight, 1.0, 0.02)
        init.constant_(mod.bias, 0.0)

netg = netg.to(device)
netd = netd.to(device)
neta = neta.to(device)

dataset = LookbookDataset(data_dir='/home/pramati/sravya/ViewU/lookbook/data/',
                          index_dir='../tool/')

iteration = 0
lr = 0.0002
real_label = 1
fake_label = 0
fineSize = 64

label = th.zeros((128, 1), requires_grad=False).to(device)
optimG = optim.Adam(netg.parameters(), lr=lr/2)
optimD = optim.Adam(netd.parameters(), lr=lr/3)
optimA = optim.Adam(neta.parameters(), lr=lr/3)
print('Training starts')
while iteration < 1000000:
    ass_label, noass_label, img1, img2 = dataset.getbatch(128)
    ass_label = ass_label.to(device).to(th.float32)
    noass_label = noass_label.to(device).to(th.float32)
    img1 = img1.to(device).to(th.float32)
    img2 = img2.to(device).to(th.float32)
    # update D
    lossD = 0
    optimD.zero_grad()
    output = netd(ass_label)
    label.fill_(real_label)
    lossD_real1 = F.binary_cross_entropy(output, label)
    lossD += lossD_real1.item()
    lossD_real1.backward()

    label.fill_(real_label)
    output1 = netd(noass_label)
    lossD_real2 = F.binary_cross_entropy(output1, label)
    lossD == lossD_real2.item()
    lossD_real2.backward()

    fake = netg(img1).detach()
    label.fill_(fake_label)
    output2 = netd(fake)

    lossD_fake1 = F.binary_cross_entropy(output2, label)
    lossD += lossD_fake1.item()
    lossD_fake1.backward()

    fake = netg(img2).detach()
    label.fill_(fake_label)
    output3 = netd(fake)

    lossD_fake2 = F.binary_cross_entropy(output3, label)
    lossD += lossD_fake2.item()
    lossD_fake2.backward()

    optimD.step()
    # update A
    lossA = 0
    optimA.zero_grad()
    assd1 = th.cat((img1, ass_label), 1)
    assd2 = th.cat((img2, ass_label), 1)
    noassd1 = th.cat((img1, noass_label), 1)
    noassd2 = th.cat((img2, noass_label), 1)
    fake1 = netg(img1).detach()
    fake2 = netg(img2).detach()
    faked1 = th.cat((img1, fake1), 1)
    faked2 = th.cat((img2, fake2), 1)

    label.fill_(real_label)
    output1 = neta(assd1)
    lossA_real1 = F.binary_cross_entropy(output1, label)
    lossA += lossA_real1.item()
    lossA_real1.backward()

    label.fill_(real_label)
    output2 = neta(assd2)
    lossA_real2 = F.binary_cross_entropy(output2, label)
    lossA += lossA_real2.item()
    lossA_real2.backward()

    label.fill_(fake_label)
    output3 = neta(noassd1)
    lossA_real3 = F.binary_cross_entropy(output3, label)
    lossA += lossA_real3.item()
    lossA_real3.backward()

    label.fill_(fake_label)
    output4 = neta(noassd2)
    lossA_real4 = F.binary_cross_entropy(output4, label)
    lossA += lossA_real4.item()
    lossA_real4.backward()

    label.fill_(fake_label)
    output5 = neta(faked1)
    lossA_fake1 = F.binary_cross_entropy(output5, label)
    lossA += lossA_fake1.item()
    lossA_fake1.backward()

    label.fill_(fake_label)
    output6 = neta(faked2)
    lossA_fake2 = F.binary_cross_entropy(output6, label)
    lossA += lossA_fake2.item()
    lossA_fake2.backward()
    optimA.step()
    # update G
    lossG = 0
    optimG.zero_grad()
    fake1 = netg(img1)
    output1 = netd(fake1)

    label.fill_(real_label)
    lossGD = F.binary_cross_entropy(output1, label)
    lossG += lossGD.item()
    lossGD.backward(retain_graph=True)

    fake2 = netg(img2)
    output2 = netd(fake2)

    label.fill_(real_label)
    lossGD = F.binary_cross_entropy(output2, label)
    lossG += lossGD.item()
    lossGD.backward(retain_graph=True)

    faked1 = th.cat((img1, fake1), 1)
    output3 = neta(faked1)
    label.fill_(real_label)
    lossGA = F.binary_cross_entropy(output3, label)
    lossG += lossGA.item()
    lossGA.backward()

    faked2 = th.cat((img2, fake2), 1)
    output4 = neta(faked2)
    label.fill_(real_label)
    lossGA = F.binary_cross_entropy(output4, label)
    lossG += lossGA.item()
    lossGA.backward()
    optimG.step()

    iteration += 1

    if iteration % 20 == 0:
        with th.no_grad():
            netg.eval()
            fake1 = netg(img1)
            fake2 = netg(img2)
            netg.train()
        fake1 = (fake1 + 1) / 2 * 255
        fake2 = (fake2 + 1) / 2 * 255
        real = (ass_label + 1) / 2 * 255
        ori1 = (img1 + 1) / 2 * 255
        ori2 = (img2 + 1) / 2 * 255
        al = th.cat((fake1, fake2, real, ori1, ori2), 2)
        display = make_grid(al, 10).cpu().numpy()
        if win1 is None:
            win1 = vis.image(display, opts=dict(title="fake", caption='fake'))
        else:
            vis.image(display, win=win1)
    if iteration % 20 == 0:
        print('iter = {}, ErrG = {}, ErrA = {}, ErrD = {}'.format(
            iteration, lossG/2, lossA/3, lossD/3
        ))
        if win is None:
            win = vis.line(X=np.array([[iteration, iteration,
                                        iteration]]),
                           Y=np.array([[lossG/2, lossA/3, lossD/3]]),
                           opts=dict(
                               ylabel='loss',
                               xlabel='iterations',
                               legend=['lossG', 'lossA', 'lossD']
                           ))
        else:
            vis.line(X=np.array([[iteration, iteration,
                                  iteration]]),
                     Y=np.array([[lossG/2, lossA/3, lossD/3]]),
                     win=win,
                     update='append')
