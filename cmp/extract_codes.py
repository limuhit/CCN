import numpy
import lmdb
import numpy as np
import os
import caffe
def ch_files_by_idx(nchannel):
    f = open('./model/35_base_extractor.prototxt')
    res=''
    for pt in f.readlines():
        if pt.find('channels')>=0:
            res+='	  channels: %d\n'%nchannel
        else:
            res+=pt
    f.close()
    f = open('./model/35_base_extractor.prototxt','w')
    f.write(res)
    f.close()
def create_lmdb_code(net,num,model,prex='train'):
    shape=net.blobs['out'].data.shape[1:]
    X=np.zeros(shape,dtype=np.uint8)
    map_size=X.nbytes * num *1.6
    env = lmdb.open('f:/compress/code_full_%s_%d_lmdb'%(prex,model), map_size)
    i = 0
    datum=caffe.proto.caffe_pb2.Datum()
    datum.channels=shape[0]
    datum.height=shape[1]
    datum.width=shape[2]
    with env.begin(write=True) as txn:
        for pid in range(num):
            net.forward()
            datum.data=net.blobs['out'].data[0].astype(np.uint8).tobytes()
            datum.label=int(pid)
            stri_id='{:06}'.format(i)
            i = i+1
            txn.put(stri_id.encode('ascii'),datum.SerializePartialToString())
            if i % 100 == 0:
               print i  
if __name__ == '__main__':
    train_num=40000
    test_num=24
    caffe.set_device(0)
    caffe.set_mode_gpu()
    model = './model/gmm/8.caffemodel'
    nchannels = 8
    model_idx = 82
    train = True
    ch_files_by_idx(nchannels)
    train = False
    if train:
        net=caffe.Net('./model/35_base_extractor.prototxt',model,caffe.TRAIN)
        create_lmdb_code(net,train_num,model_idx,'train')
    else:
        net=caffe.Net('./model/35_base_extractor.prototxt',model,caffe.TEST)
        create_lmdb_code(net,test_num,model_idx,'test')