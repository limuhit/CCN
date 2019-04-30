import caffe
import numpy as np
import os
if __name__ == '__main__':
    gpu_id=1
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    solver=caffe.AdamSolver('./model/lossless_adam_solver.prototxt')
    #solver.restore('./model/save/ls_14_iter_40000.solverstate')
    iters=100000
    for i in range(iters):
        solver.step(10)
        print 'gpu_id:%d, entropy:%.3f'%(gpu_id,
                                         solver.net.blobs['ent_loss'].data)
        
    
