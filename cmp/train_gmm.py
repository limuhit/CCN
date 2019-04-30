import caffe
import numpy as np
def test(net):
    pmse = 0
    pent = 0
    for i in range(24):
        net.forward()
        pmse += (net.blobs['loss'].data+0)
        pent += (net.blobs['ent_loss'].data+0)
    print pmse/24.0, pent/24.0 
if __name__ == '__main__':
    gpu_id=0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    solver=caffe.AdamSolver('./model/cmp_adam_solver.prototxt')
    solver.net.copy_from('./model/save/anchor/base_20_sim.caffemodel')
    solver.net.copy_from('./model/save/anchor/base_20_sim_entropy.caffemodel')
    solver.test_nets[0].copy_from('./model/save/anchor/base_20_sim.caffemodel')
    solver.test_nets[0].copy_from('./model/save/anchor/base_20_sim_entropy.caffemodel')
    #solver.restore('./model/save/gmm_20_iter_150000.solverstate')
    iters=100000
    for i in range(iters):
        #if i % 50 == 0: test(solver.test_nets[0])
        solver.step(10)
        print 'gpu_id:%d, l2:%.4f, ms-ssim:%.4f,entropy:%.3f'%(gpu_id,solver.net.blobs['loss'].data,
                                                  solver.net.blobs['mloss'].data,
                                                  solver.net.blobs['ent_loss'].data)
        
    
