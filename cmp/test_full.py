import caffe
import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
import pickle

def get_flist(name):
    f=open(name)
    flist=[pt[:-1] for pt in f.readlines()]
    f.close()
    return flist

def test_encoder(net,img_dir,working_dir):
    flist = get_flist(img_dir)
    for idx,pt in zip(range(1,len(flist)+1),flist):
        img=cv2.imread(pt)
        img=img.transpose(2,0,1).astype(np.float32)
        net.blobs['data'].reshape(1,3,img.shape[1],img.shape[2])
        net.blobs['data'].data[...]=img
        net.reshape()
        net.forward()
        np.save('./%s/%d_quant.npy'%(working_dir,idx),net.blobs['tdata_out'].data[0])
        np.save('./%s/%d_int.npy'%(working_dir,idx),net.blobs['tdata_int'].data[0])

def test_decoder(net,img_dir,working_dir,out_dir='./tec/msssim'):
    spr=0
    shown=False
    psnr=lambda x,y: 10*math.log10(255*255/np.average(np.square(x.astype(np.float)-y)))
    sim=0
    srt=0
    nchannel=net.blobs['tdata_int_sc'].shape[1]
    flist = get_flist(img_dir)
    rt_list=[]
    pr_list=[]
    sim_list=[]
    for i in xrange(1,len(flist)+1):
        img=cv2.imread(flist[i-1])
        data=img.transpose(2,0,1).astype(np.float32)
        net.blobs['data'].reshape(1,3,data.shape[1],data.shape[2])
        net.blobs['data'].data[...]=data
        quant=np.load('./%s/%d_quant.npy'%(working_dir,i))
        tint=np.load('./%s/%d_int.npy'%(working_dir,i))
        net.blobs['tdata_out'].reshape(1,quant.shape[0],quant.shape[1],quant.shape[2])
        net.blobs['tdata_out'].data[0]=quant.astype(np.float32)
        net.blobs['tdata_int'].reshape(1,tint.shape[0],tint.shape[1],tint.shape[2])
        net.blobs['tdata_int'].data[0]=tint.astype(np.float32)
        net.reshape()
        net.forward()
        #draw(net)
        gdata = net.blobs['gdata_scale'].data[0]+0.5
        gdata[gdata<0]=0
        gdata[gdata>255]=255
        gdata=gdata.transpose(1,2,0).astype(np.uint8)
        pr = psnr(img,gdata)
        pr_list.append(pr+0)
        sim_list.append(net.blobs['mloss'].data+0)
        spr+=pr
        srt+=(net.blobs['ent_loss'].data+0)
        sim+=(net.blobs['mloss'].data+0)
        rt=int(nchannel*(net.blobs['ent_loss'].data+0)/0.693/64.0*1000)
        rt_list.append(rt/1000.0)
        image_name = '%d_%s.png'%(i,'{:05}'.format(rt))
        cv2.imwrite(os.path.join(out_dir,image_name),gdata)
        #print pr
        
    print  spr/100.0,nchannel*srt/100.0/0.693/64.0,sim/100.0
    return rt_list,pr_list,sim_list

if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    model_list=[4,8,14,20,26,32]
    model_list=[4,6,8,14,20,26,32]
    img_dir='E:/data/name2.txt'
    idx = model_list[5]
    model = './model/save/gmm/%d_sim.caffemodel'%idx
    encoder_prototxt = './model/gmm_%d_deploy_encoder.prototxt'%idx
    decoder_prototxt = './model/gmm_%d_deploy_decoder.prototxt'%idx
    encoding = True
    encoding = False
    if encoding:  encoder = caffe.Net(encoder_prototxt,model,caffe.TEST)
    else: decoder = caffe.Net(decoder_prototxt,model,caffe.TEST)
    if encoding:
        test_encoder(encoder,img_dir,'./tmp')
    else:
        rt,pr,sim = test_decoder(decoder,img_dir,'./tmp','./tec/msssim')
        f = open('./model/save/gmm/%d_sim.txt'%idx,'w')
        tmp = ['%.3f,%.3f,%.3f\n'%(pa,pb,pc) for pa,pb,pc in zip(rt,pr,sim)]
        res = reduce(lambda xa,xb:xa+xb,tmp)
        f.write(res)
        f.close()
    