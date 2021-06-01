#coding=utf-8
import caffe
import numpy as np
import pickle

if __name__ == '__main__':
    caffe.set_mode_cpu()
    model_idx_list = [4]#[4,8,14]
    prex_list = ['']#['','sim']
    for model_idx in model_idx_list:
        for prex in prex_list:
            #net = './model/decoder_%d.prototxt'%model_idx
            #net = './model/decoder.prototxt'
            net = './model/lossless_{}_encoder.prototxt'.format(model_idx)
            if prex == '':
                model = './model/code_%d.caffemodel' % model_idx
            else:
                model = './model/code_%d_sim.caffemodel' % model_idx
            out_dir = 'g:/caffe_to_torch'
            net = caffe.Net(net,model,caffe.TEST)
            param = {}
            for pname in net.params.keys():
                current_vec = []
                for idx in range(len(net.params[pname])):
                    current_vec.append(net.params[pname][idx].data)
                param[pname] = current_vec
            '''
            if prex == '':
                out_name = '%s/%d_mse_encoder.data'%(out_dir,model_idx)
            else:
                out_name = '%s/%d_sim_encoder.data'%(out_dir,model_idx)
            with open(out_name,'wb') as f:
                pickle.dump(param, f, protocol=pickle.HIGHEST_PROTOCOL)
            #'''
            print(net.params.keys())
