import caffe
import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
def draw(net):
    pi = 1./math.sqrt(2*math.acos(-1.0))
    gs = lambda x,dx,mx : pi/ dx * math.exp(-(x-mx)*(x-mx)/(2.*dx*dx))
    idx=np.random.randint(0,net.blobs['ent_delta_gdata_tans'].data.shape[0],100)
    for i in range(100):
        delta = net.blobs['ent_delta_gdata_tans'].data[idx[i]]
        mean = net.blobs['ent_mean_gdata_tans'].data[idx[i]]
        wt = net.blobs['ent_weight_softmax'].data[idx[i]]
        x = np.linspace(-4,4,200)
        y = [wt[0]*gs(pt,delta[0],mean[0])+wt[1]*gs(pt,delta[1],mean[1])+wt[2]*gs(pt,delta[2],mean[2]) for pt in x]
        plt.figure()
        plt.plot(x+4.0,y)
        plt.show()

def test(net,img_dir,out_dir='./img/mssim'):
    spr=0
    shown=False
    psnr=lambda x,y: 10*math.log10(255*255/np.average(np.square(x.astype(np.float)-y)))
    sim=0
    srt=0
    nchannel=net.blobs['tdata_int_sc'].shape[1]
    for i in xrange(1,25):
        img=cv2.imread('%s/%d.png'%(img_dir,i))
        h,w=img.shape[:2]
        net.blobs['data'].reshape(1,3,h,w)
        net.blobs['data'].data[0]=img.transpose(2,0,1).astype(np.float32)
        net.forward()
        #draw(net)
        gdata = net.blobs['gdata_scale'].data[0]+0.5
        gdata[gdata<0]=0
        gdata[gdata>255]=255
        gdata=gdata.transpose(1,2,0).astype(np.uint8)
        pr = psnr(img,gdata)
        spr+=pr
        srt+=(net.blobs['ent_loss'].data+0)
        sim+=(net.blobs['mloss'].data+0)
        rt=int(nchannel*(net.blobs['ent_loss'].data+0)/0.693/64.0*1000)
        image_name = '%d_%s.png'%(i,'{:05}'.format(rt))
        cv2.imwrite(os.path.join(out_dir,image_name),gdata)
        #print pr
    return  spr/24.0,nchannel*srt/24.0/0.693/64.0,sim/24.0, srt/24.0

if __name__ == '__main__':
    caffe.set_device(1)
    caffe.set_mode_gpu()
    model_list=[4,8,14,20,26,32]
    #model_list=[4,6,8,14,20,26,32]
    model_list=[8]
    img_dir='E:/data/images/whole/'
    model = './model/save/gmm/%d.caffemodel'
    prototxt = './model/gmm_%d_deploy.prototxt'
    res=''
    lrt=[]
    lsim=[]
    lpr=[]
    for pt in model_list:
        net = caffe.Net(prototxt%pt, model%pt, caffe.TEST)
        pr,rt,sim,ent=test(net,img_dir,'./img/tmp')
        print ent
        lrt.append(round(rt,3))
        lsim.append(round(sim,4))    
        lpr.append(round(pr,3))
        res += 'psnr:%.3f, ms-sim:%.4f, rt:%.3f\n'%(pr,sim, rt)
    print res
    print lrt
    print lsim
    print lpr
    