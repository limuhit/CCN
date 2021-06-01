import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import cv2
import numpy as np
from CCN_operator import QUANT, Dquant, Dtow, MaskData, DExtract, DInput, DOutput, ConstScale, DConv
from enum import Enum
import coder
import struct
import os

class NetType(Enum):
    ENC = 1
    DEC = 2
    ENT_ENC = 3
    ENT_DEC = 4

def load_dicts(key, code_channel):
    
    if key == NetType.ENC:
        encoder_dict = OrderedDict([
            ('down1', nn.Conv2d(3,64,3,2,1)),
            ('conv1', nn.Conv2d(64,64,3,1,1)),
            ('conv1_PReLU', nn.PReLU(64)),
            ('conv2', nn.Conv2d(64,64,3,1,1)),
            ('conv2_PReLU', nn.PReLU(64)),
            ('conv3', nn.Conv2d(64,64,3,1,1)),
            ('conv3_PReLU', nn.PReLU(64)),
            ('concat1', 'conv1_PReLU'),
            ('conv4', nn.Conv2d(128,64,3,1,1)),
            ('conv4_PReLU', nn.PReLU(128)),
            ('conv5', nn.Conv2d(64,64,3,1,1)),
            ('conv5_PReLU', nn.PReLU(64)),
            ('concat2', 'concat1'),
            ('conv6', nn.Conv2d(192,64,3,1,1)),
            ('conv6_PReLU', nn.PReLU(192)),
            ('conv7', nn.Conv2d(64,64,3,1,1)),
            ('conv7_PReLU', nn.PReLU(64)),
            ('concat3', 'concat2'),
            ('down2', nn.Conv2d(256,128,3,2,1)),
            ('down2_PReLU', nn.PReLU(256)),
            ('conv8', nn.Conv2d(128,128,3,1,1)),
            ('conv8_PReLU', nn.PReLU(128)),
            ('conv9', nn.Conv2d(128,64,3,1,1)),
            ('conv9_PReLU', nn.PReLU(128)),
            ('conv10', nn.Conv2d(64,64,3,1,1)),
            ('conv10_PReLU', nn.PReLU(64)),
            ('concat4', 'conv8_PReLU'),
            ('conv11', nn.Conv2d(192,64,3,1,1)),
            ('conv11_PReLU', nn.PReLU(192)),
            ('conv12', nn.Conv2d(64,64,3,1,1)),
            ('conv12_PReLU', nn.PReLU(64)),
            ('concat5', 'concat4'),
            ('conv13', nn.Conv2d(256,64,3,1,1)),
            ('conv13_PReLU', nn.PReLU(256)),
            ('conv14', nn.Conv2d(64,64,3,1,1)),
            ('conv14_PReLU', nn.PReLU(64)),
            ('concat6', 'concat5'),
            ('down3', nn.Conv2d(320,256,3,2,1)),
            ('down3_PReLU', nn.PReLU(320)),
            ('conv15', nn.Conv2d(256,256,3,1,1)),
            ('conv15_PReLU', nn.PReLU(256)),
            ('conv16', nn.Conv2d(256,128,3,1,1)),
            ('conv16_PReLU', nn.PReLU(256)),
            ('conv17', nn.Conv2d(128,128,3,1,1)),
            ('conv17_PReLU', nn.PReLU(128)),
            ('concat7', 'conv15_PReLU'),
            ('conv18', nn.Conv2d(384,128,3,1,1)),
            ('conv18_PReLU', nn.PReLU(384)),
            ('conv19', nn.Conv2d(128,128,3,1,1)),
            ('conv19_PReLU', nn.PReLU(128)),
            ('concat8', 'concat7'),
            ('conv20', nn.Conv2d(512,128,3,1,1)),
            ('conv20_PReLU', nn.PReLU(512)),
            ('conv21', nn.Conv2d(128,128,3,1,1)),
            ('conv21_PReLU', nn.PReLU(128)),
            ('concat9', 'concat8'),
            ('conv_encoder', nn.Conv2d(640,512,3,1,1)),
            ('conv_encoder_PReLU', nn.PReLU(640)),
            ('Tdata', nn.Conv2d(512,32,3,1,1)),
            ('Tdata_Sigmoid', nn.Sigmoid()),
            ('tdata_quant',QUANT(32,8,ntop=2))])
        return encoder_dict
    elif key == NetType.DEC:
        decoder_dict=OrderedDict([
        ('tdata_quant', Dquant(32,8)),
        ('mask_data',MaskData(code_channel,True)),
        ('conv_decoder',nn.Conv2d(32,512,3,1,1)),
        ('conv_decoder_PReLU',nn.PReLU(512)), 
        ('iconv1', nn.Conv2d(512,256,3,1,1)), 
        ('iconv1_PReLU',nn.PReLU(256)), 
        ('iconv2', nn.Conv2d(256,128,3,1,1)), 
        ('iconv2_PReLU',nn.PReLU(128)),
        ('iconv3', nn.Conv2d(128,128,3,1,1)), 
        ('iconv3_PReLU',nn.PReLU(128)),
        ('concat1','iconv1_PReLU'),
        ('iconv4', nn.Conv2d(384,128,3,1,1)), 
        ('iconv4_PReLU',nn.PReLU(128)),
        ('iconv5', nn.Conv2d(128,128,3,1,1)), 
        ('iconv5_PReLU',nn.PReLU(128)),
        ('concat2','concat1'),
        ('iconv6', nn.Conv2d(512,128,3,1,1)), 
        ('iconv6_PReLU',nn.PReLU(128)),
        ('iconv7', nn.Conv2d(128,128,3,1,1)), 
        ('iconv7_PReLU',nn.PReLU(128)),
        ('concat3','concat2'),
        ('iconv_up1', nn.Conv2d(640,256,3,1,1)), 
        ('iconv_up1_PReLU',nn.PReLU(256)),
        ('up1',Dtow(2,True,0)),
        ('iconv8', nn.Conv2d(64,128,3,1,1)), 
        ('iconv8_PReLU',nn.PReLU(128)),
        ('iconv9', nn.Conv2d(128,64,3,1,1)), 
        ('iconv9_PReLU',nn.PReLU(64)),
        ('iconv10', nn.Conv2d(64,64,3,1,1)), 
        ('iconv10_PReLU',nn.PReLU(64)),
        ('concat4','iconv8_PReLU'),
        ('iconv11', nn.Conv2d(172,64,3,1,1)), 
        ('iconv11_PReLU',nn.PReLU(64)),
        ('iconv12', nn.Conv2d(64,64,3,1,1)), 
        ('iconv12_PReLU',nn.PReLU(64)),
        ('concat5','concat4'),
        ('iconv13', nn.Conv2d(256,64,3,1,1)), 
        ('iconv13_PReLU',nn.PReLU(64)),
        ('iconv14', nn.Conv2d(64,64,3,1,1)), 
        ('iconv14_PReLU',nn.PReLU(64)),
        ('concat6','concat5'),
        ('iconv_up2', nn.Conv2d(320,128,3,1,1)), 
        ('iconv_up2_PReLU',nn.PReLU(128)),
        ('up2',Dtow(2,True,0)),
        ('iconv15', nn.Conv2d(32,64,3,1,1)), 
        ('iconv15_PReLU',nn.PReLU(64)),
        ('iconv16', nn.Conv2d(64,64,3,1,1)), 
        ('iconv16_PReLU',nn.PReLU(64)),
        ('iconv17', nn.Conv2d(64,64,3,1,1)), 
        ('iconv17_PReLU',nn.PReLU(64)),
        ('concat7','iconv15_PReLU'),
        ('iconv18', nn.Conv2d(128,64,3,1,1)), 
        ('iconv18_PReLU',nn.PReLU(64)),
        ('iconv19', nn.Conv2d(64,64,3,1,1)), 
        ('iconv19_PReLU',nn.PReLU(64)),
        ('concat8','concat7'),
        ('iconv20', nn.Conv2d(192,64,3,1,1)), 
        ('iconv20_PReLU',nn.PReLU(64)),
        ('iconv21', nn.Conv2d(64,64,3,1,1)), 
        ('iconv21_PReLU',nn.PReLU(64)),
        ('concat9','concat8'),
        ('iconv_up3', nn.Conv2d(256,64,3,1,1)), 
        ('iconv_up3_PReLU',nn.PReLU(64)),
        ('up3',Dtow(2,True,0)),
        ('conv_dataA', nn.Conv2d(16,64,3,1,1)), 
        ('conv_dataA_PReLU',nn.PReLU(64)),
        ('conv_dataB', nn.Conv2d(64,64,3,1,1)), 
        ('conv_dataB_PReLU',nn.PReLU(64)),
        ('conv_dataC', nn.Conv2d(64,64,3,1,1)), 
        ('conv_dataC_PReLU',nn.PReLU(64)),
        ('concat10','up3'),
        ('gdata', nn.Conv2d(80,3,3,1,1))
        ])
        return decoder_dict
    elif key == NetType.ENT_ENC or key == NetType.ENT_DEC:
        ent_encoder_dict = OrderedDict([
            ('rdata',DInput(code_channel)),
            ('gdata', ConstScale(0.125,0)),
            ('conv1', DConv(code_channel,1,8,5,False)),
            ('conv1_relu', nn.PReLU(code_channel*8)),
            ('conv2', DConv(code_channel,8,8,5,True)),
            ('conv2_relu', nn.PReLU(code_channel*8)),
            ('ent_blk1_1', DConv(code_channel,8,8,5,True)),
            ('ent_blk1_1_relu', nn.PReLU(code_channel*8)),
            ('ent_blk1_2', DConv(code_channel,8,8,5,True)),
            ('ent_blk1_2_relu', nn.PReLU(code_channel*8)),
            ('add1', 'conv2_relu'),
            ('ent_blk2_1', DConv(code_channel,8,8,5,True)),
            ('ent_blk2_1_relu', nn.PReLU(code_channel*8)),
            ('ent_blk2_2', DConv(code_channel,8,8,5,True)),
            ('ent_blk2_2_relu', nn.PReLU(code_channel*8)),
            ('add2', 'add1'),
            ('ent_blk3_1', DConv(code_channel,8,8,5,True)),
            ('ent_blk3_1_relu', nn.PReLU(code_channel*8)),
            ('ent_blk3_2', DConv(code_channel,8,8,5,True)),
            ('ent_blk3_2_relu', nn.PReLU(code_channel*8)),
            ('add3', 'add2'),
            ('pdata', DConv(code_channel,8,8,3,True)),
            ('pred', DOutput(code_channel))
            ])
        return ent_encoder_dict

class Net(nn.Module):
    def __init__(self, net_type, code_channels=4, prex='mse'):
        super(Net, self).__init__()
        self.model = load_dicts(net_type, code_channels)
        self.concat_dic = {}
        self.path = 'g:/caffe_to_torch/'
        self.nt = net_type
        self.prex = prex
        self.code_channels = code_channels
        for pkey in self.model.keys():
            if pkey.find('concat')>=0 or pkey.find('add')>=0:
                self.concat_dic[self.model[pkey]] = None
            elif pkey == 'rdata':
                self.concat_dic[pkey] = None
        #print(self.concat_dic)

    def to(self, device):
        for pkey in self.model.keys():
            if pkey.find('concat')<0 and pkey.find('add')<0: 
                self.model[pkey]=self.model[pkey].to(device)
        return self

    def forward(self, x):
        for pkey in self.model.keys():
            if pkey.find('concat')>=0:
                x = torch.cat([self.concat_dic[self.model[pkey]],x], dim=1)
            elif pkey.find('add')>=0:
                x = x + self.concat_dic[self.model[pkey]]
            else:
                x = self.model[pkey](x)
            if pkey in self.concat_dic.keys():
                self.concat_dic[pkey] = x
        if self.nt==NetType.ENT_DEC:
            return x[0], x[1], self.concat_dic['rdata']
        else:
            return x
    
    @staticmethod
    def get_params(fname):
        with open(fname, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load() 
        return p

    @staticmethod
    def set_params(model,param):
        for pkey in model.keys():
            if isinstance(model[pkey], nn.Conv2d) or isinstance(model[pkey],DConv):
                model[pkey].weight = nn.Parameter(torch.from_numpy(param[pkey][0]))
                model[pkey].bias = nn.Parameter(torch.from_numpy(param[pkey][1]).view(-1))
            elif isinstance(model[pkey], nn.PReLU):
                model[pkey].weight = nn.Parameter(torch.from_numpy(param[pkey][0]).view(-1))
            elif pkey.find('quant')>=0: 
                model[pkey].weight = nn.Parameter(torch.from_numpy(param[pkey][0]).view(model[pkey].weight.shape))

    def load_params(self):

        if self.nt == NetType.ENC:
            fn = '{}{}_{}_encoder.data'.format(self.path, self.code_channels, self.prex)
        elif self.nt == NetType.DEC:
            fn = '{}{}_{}_decoder.data'.format(self.path, self.code_channels, self.prex)
        else:
            fn = '{}lossless_{}_{}.data'.format(self.path, self.code_channels, self.prex)

        param = Net.get_params(fn)
        Net.set_params(self.model,param)
        
class EntEncoder(nn.Module):

    def __init__(self, fname = './tmp/test.data', code_channels=4, prex='mse'):
        super(EntEncoder,self).__init__()
        self.net = Net(NetType.ENT_ENC,code_channels,prex)
        self.ext_data = DExtract(False)
        self.ext_label = DExtract(True)
        self.mcoder = coder.coder(fname)

    def to(self,device):
        self.net = self.net.to(device)
        self.ext_data = self.ext_data.to(device)
        self.ext_label = self.ext_label.to(device)
        return self

    def load_params(self):
        self.net.load_params()

    def forward(self,x):
        _,c,h,w = x.shape
        self.mcoder.start_encoder()
        for _ in range(c+h+w-2):
            tx = self.ext_data(x)
            pred,num = self.net(tx)
            tlabel = self.ext_label(x)
            self.mcoder.encodes(pred,8,tlabel,int(num[0].item()))
        self.mcoder.end_encoder()
    
class EntDecoder(nn.Module):
    
    def __init__(self, fname = './tmp/test.data', code_channels=4, prex='mse'):
        super(EntDecoder,self).__init__()
        self.net = Net(NetType.ENT_DEC,code_channels,prex)
        self.mcoder = coder.coder(fname)
        self.code_channels = code_channels

    def to(self,device):
        self.net = self.net.to(device)
        return self

    def load_params(self):
        self.net.load_params()

    def forward(self,n,h,w,device):
        pout = torch.zeros((n,1,h,w),dtype=torch.float32).to(device)
        self.mcoder.start_decoder()
        for _ in range(self.code_channels+h+w-2):
            pred,num,rdata = self.net(pout)
            pout = self.mcoder.decodes(pred,8,int(num[0].item())).to(device).view(n,1,h,w).contiguous()
        rdata[:,-1,-1,-1] = pout[:,0,0,0]
        return rdata

def save_shape(h,w,fname='test.shape'):
    byte = h*256*256+w
    f = open(fname,'wb')
    f.write(struct.pack('I',byte))
    f.close()

def load_shape(fname='test.shape'):
    f = open(fname,'rb')
    res = f.read(4)
    num = struct.unpack('I',res)[0]
    f.close()
    mod = 256*256
    h = num // mod
    w = num % mod
    return h,w

def encoding(img_name, code_name, model_idx=0, prex='mse'):
    model_list = [4,8,14]
    code_channels = model_list[model_idx]
    device = torch.device("cuda:0")
    with torch.no_grad():
        net = Net(NetType.ENC,code_channels,prex)
        net.load_params()
        net = net.to(device)
        img = cv2.imread(img_name)
        h,w = img.shape[:2]
        th,tw = h//8*8,w//8*8
        ph, pw = (h-th)//2, (w-tw)//2
        img = img[ph:ph+th, pw:pw+tw]
        img = torch.from_numpy(img.transpose(2,0,1)).type(torch.float32).contiguous()
        img = img / 255.0
        data = img.view(1,3,th,tw).to(device)
        _,y2 = net(data)
        ent_enc = EntEncoder(code_name,code_channels,prex)
        ent_enc.load_params()
        ent_enc = ent_enc.to(device)
        ent_enc.forward(y2[:,:code_channels].contiguous())
        fshape = code_name + '.shape'
        save_shape(th//8,tw//8,fshape)
        sz = os.path.getsize(code_name)
    print('finish encoding {}, total_bits: {}, bitrate: {:.3}bpp'.format(img_name, sz,float(sz)*8./th/tw))   


def decoding(code_name, model_idx=0, prex='mse'):
    model_list = [4,8,14]
    code_channels = model_list[model_idx]
    device = torch.device("cuda:0")
    fshape = code_name + '.shape'
    h,w = load_shape(fshape)
    with torch.no_grad():
        ent_dec = EntDecoder(code_name,code_channels,prex)
        ent_dec.load_params()
        ent_dec = ent_dec.to(device)
        #ent_enc.forward(y2[:,:14].contiguous())
        y3=ent_dec.forward(1,h,w,device)
        y2=torch.zeros((1,32,h,w),dtype=torch.float32,device=device)
        y2[:,:code_channels] = y3
        net = Net(NetType.DEC,code_channels,prex)
        net.load_params()
        net = net.to(device)
        z = net(y2)
        z[z<0]=0
        z[z>1]=1
        dimg = z[0].detach().to('cpu').numpy().transpose(1,2,0)*255
        dimg = dimg.astype(np.uint8)
        cv2.imshow('dimg',dimg)
        cv2.waitKey()

if __name__ == '__main__':
    img_name = 'H:/image_test_set/kodak/2.png'
    encoding(img_name,'./tmp/a.data',2,'mse')
    decoding('./tmp/a.data',2,'mse')