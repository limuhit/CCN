import caffe
import numpy as np
import os
if __name__ == '__main__':
    gpu_id=1
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    solver=caffe.AdamSolver('./model/cmp_adam_base_solver.prototxt')
    #solver.net.copy_from('./model/save/anchor/base_14_mse.caffemodel')
    solver.net.params['tdata_mask'][0].data[0,0,0,0]=4.0
    #solver.restore('./model/save/base_4_iter_60000.solverstate')
    iters=100000
    for i in range(iters):
        solver.step(10)
        print 'gpu_id:%d, l2:%.4f, ms-ssim:%.4f'%(gpu_id,solver.net.blobs['loss'].data,
                                                  solver.net.blobs['mloss'].data)
        
    
