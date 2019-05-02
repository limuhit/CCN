import argparse
import Coder
import caffe
import numpy as np
import cv2
import os
import time          
import struct
def encoding(net,file_name='test.data'):
    coder = Coder.coder(file_name)
    coder.start_encoder()
    ta = time.time()
    shape = net.blobs['rdata'].data.shape[1:]
    for i in range(np.sum(shape)-2):
        net.forward()
        pred = net.blobs['pred'].data.reshape(-1,9)
        label = net.blobs['label'].data.reshape(-1)
        j = 0 
        while pred[j,0]>=0:
            coder.encode(pred[j].astype(np.uint32),8,int(pred[i,8]),int(label[j]))
            j+=1
    print net.blobs['rdata'].data.size
    coder.end_encoder()
    tb = time.time()
    print tb-ta
def encoding2(net,file_name='test.data'):
    coder = Coder.coder(file_name)
    coder.start_encoder()
    ta = time.time()
    shape = net.blobs['rdata'].data.shape[1:]
    for i in range(np.sum(shape)-2):
        net.forward()
        pred = net.blobs['pred'].data.reshape(-1).astype(np.uint32)
        label = net.blobs['label'].data.reshape(-1).astype(np.uint32)
        coder.encodes(pred,8,label)
    print net.blobs['rdata'].data.size
    coder.end_encoder()
    tb = time.time()
    print tb-ta
def decoding(net,file_name='test.data'):
    decoder =  Coder.coder(file_name)
    decoder.start_decoder()
    ta = time.time()
    shape = net.blobs['rdata'].data.shape[1:]
    data = np.zeros(shape[1]*shape[2]).astype(np.float32)
    for i in range(np.sum(shape)-2):
        net.forward()
        pred = net.blobs['pred'].data.reshape(-1,9)
        j = 0 
        while pred[j,0]>=0:
            res=decoder.decode(pred[j].astype(np.uint32),8,int(pred[i,8]))
            data[j]=res
            j+=1
        net.blobs['data'].data[0,0]=data.reshape(shape[1],shape[2])
    rcode = net.blobs['rdata'].data[0]
    rcode[-1,-1,-1]=data[0]
    #print np.sum(np.fabs(code-net.blobs['rdata'].data[0]))
    return rcode
def decoding2(net,file_name='test.data'):
    decoder =  Coder.coder(file_name)
    decoder.start_decoder()
    ta = time.time()
    shape = net.blobs['rdata'].data.shape[:]
    data = np.zeros(shape[0]*shape[2]*shape[3]).astype(np.float32)
    for i in range(np.sum(shape[1:])-2):
        net.forward()
        pred = net.blobs['pred'].data.reshape(-1).astype(np.uint32)
        decoder.decodes(pred,8,data)
        net.blobs['data'].data[...]=data.reshape(shape[0],1,shape[2],shape[3])
        #print data
    rcode = net.blobs['rdata'].data[:]
    rcode[:,-1,-1,-1]=data[:shape[0]]
    #print np.sum(np.fabs(code-net.blobs['rdata'].data[0]))
    return rcode
def cut_to_patch(data,h_m=2,w_m=2):
    c,h,w=data.shape[:]
    h_s = h//h_m
    w_s = w//w_m
    res=np.zeros((h_m*w_m,c,h_s,w_s))
    for i in range(h_m):
        for j in range(w_m):
            res[i*w_m+j]=data[:,h_s*i:h_s*i+h_s,w_s*j:w_s*j+w_s]
    return res
def merge_patchs(data,h_m=2,w_m=2):
    c,h,w=data.shape[1:]
    h_s = h*h_m
    w_s = w*w_m
    res = np.zeros((c,h_s,w_s))
    for i in range(h_m):
        for j in range(w_m):
            res[:,h*i:h*i+h,w*j:w*j+w]=data[i*w_m+j]
    return res
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
    h = num / mod
    w = num % mod
    return h,w
def pad_image(img,stride):
    h,w = img.shape[:2]
    th = (h + stride - 1) // stride * stride
    tw = (w + stride - 1) // stride * stride
    pad_h = th - h
    pad_w = tw - w
    res = np.zeros((th,tw,3))
    res[:h,:w] = img
    res[h:,:] = res[h-pad_h:h,:][::-1,:]
    res[:,w:] = res[:,w-pad_w:w][:,::-1]
    return res
def pad_shape(h,w,stride):
    th = (h + stride - 1) // stride * stride
    tw = (w + stride - 1) // stride * stride
    return th,tw
def process_whole(input_file, output_file, shape_file, encoding_flag=True):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    if encoding_flag:
        img = cv2.imread(input_file)
        sh,sw = img.shape[:2]
        img = pad_image(img,8)
        h,w = img.shape[:2]
        encoder = caffe.Net('encoder.prototxt','8.caffemodel',caffe.TEST)
        encoder.blobs['data'].reshape(1,3,h,w)
        encoder.blobs['data'].data[0]=img.transpose(2,0,1).astype(np.float32)
        encoder.reshape()
        encoder.forward()
        code = encoder.blobs['out'].data[0]
        ent2 = caffe.Net('lossless_8_encoder.prototxt', 'code_8.caffemodel',caffe.TEST)
        ent2.blobs['data'].reshape(1,code.shape[0],code.shape[1],code.shape[2])
        ent2.blobs['data'].data[0] = code
        ent2.reshape()
        encoding2(ent2,output_file)
        save_shape(sh,sw,shape_file)
    else:
        sh,sw = load_shape(shape_file)
        h,w = pad_shape(sh,sw,8)
        ent2 = caffe.Net('lossless_8_decoder.prototxt', 'code_8.caffemodel',caffe.TEST)
        ent2.blobs['data'].reshape(1,1,h//8,w//8)
        ent2.reshape()
        code = decoding2(ent2,input_file)[0]
        decoder = caffe.Net('decoder.prototxt','8.caffemodel',caffe.TEST)
        decoder.blobs['data'].reshape(1,32,h//8,w//8)
        decoder.blobs['data'].data[0,0:code.shape[0]]=code.astype(np.float32)
        decoder.reshape()
        decoder.forward()
        res = decoder.blobs['gdata_scale'].data[0]+0.5
        res[res>255]=255
        res[res<0]=0
        res = res.astype(np.uint8).transpose(1,2,0)[:sh,:sw]
        cv2.imwrite(output_file,res)
def process_patches(input_file, output_file, shape_file, encoding_flag=True):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    if encoding_flag:
        img = cv2.imread(input_file)
        h,w = img.shape[:2]
        encoder = caffe.Net('encoder.prototxt','8.caffemodel',caffe.TEST)
        encoder.blobs['data'].reshape(1,3,h,w)
        encoder.blobs['data'].data[0]=img.transpose(2,0,1).astype(np.float32)
        encoder.reshape()
        encoder.forward()
        code = encoder.blobs['out'].data[0]
        codes = cut_to_patch(code,2,2)
        ent2 = caffe.Net('lossless_8_encoder.prototxt', 'code_8.caffemodel',caffe.TEST)
        ent2.blobs['data'].reshape(*codes.shape)
        ent2.blobs['data'].data[...] = codes.astype(np.float32)
        ent2.reshape()
        encoding2(ent2,output_file)
        save_shape(h,w,shape_file)
    else:
        h,w = load_shape(shape_file)
        ent2 = caffe.Net('lossless_8_decoder.prototxt', 'code_8.caffemodel',caffe.TEST)
        ent2.blobs['data'].reshape(4,1,h//16,w//16)
        ent2.reshape()
        codes = decoding2(ent2,input_file)
        code = merge_patchs(codes,2,2)
        decoder = caffe.Net('decoder.prototxt','8.caffemodel',caffe.TEST)
        decoder.blobs['data'].reshape(1,32,h//8,w//8)
        decoder.blobs['data'].data[0,0:code.shape[0]]=code.astype(np.float32)
        decoder.reshape()
        decoder.forward()
        res = decoder.blobs['gdata_scale'].data[0]+0.5
        res[res>255]=255
        res[res<0]=0
        res = res.astype(np.uint8).transpose(1,2,0)
        cv2.imwrite(output_file,res)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', dest = 'input', 
                        help = 'For encoding, the input file is the image. For decoding, the input file is the binary file.')
    parser.add_argument('-o', '--outputfile', dest = 'output', 
                        help = 'For encoding, the output file is the binary file. For decoding, the output file is the decoding image.')
    parser.add_argument('-s', '--shape', dest = 'shape',
                        help = 'The file that stores the shape of the image.')
    parser.add_argument('-e', '--encoding', dest = 'encoding',
                        action = 'store_true', default = False, help = 'Encoding the image.')
    parser.add_argument('-p', '--patch', dest = 'patch',
                        action = 'store_true', default = False, help = 'Split the codes into patches to accelerate the entropy coding.')
    parser.add_argument('-d', '--decoding', dest = 'decoding', 
                        action = 'store_true', default = False, help  = 'Decoding the image.')
    parser.add_argument('-v', '--version', action='version',  version = '1.0')
    results = parser.parse_args()
    if results.input is None or results.output is None or results.shape is None:
        print 'Please give the inputfile, outputfile and the shapefile.'
    else:
        if results.patch:
            process_patches(results.input,results.output,results.shape,results.encoding)
        else:
            process_whole(results.input,results.output,results.shape,results.encoding)