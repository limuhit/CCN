import os
import torch

class ModuleSaver():
    def __init__(self, path='./saved_models/', prex='default'):
        self.path = path
        self.prex = prex
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.current_best_loss = None
        self.init = False
    
    def init_loss(self,loss):
        if not isinstance(loss,list): loss = [loss]
        self.current_best_loss = [pt for pt in loss]
        self.init = True
    
    def save(self, model, loss):
        res = ''
        pdict = model.module.state_dict() if isinstance(model,torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
        if not isinstance(loss,list): loss = [loss]
        if not self.init: 
            self.current_best_loss = [10e9 for _ in range(len(loss))]
            self.init = True
        flag = False
        for iter_, ploss in enumerate(loss):
            if ploss < self.current_best_loss[iter_]:
                flag = True
                self.current_best_loss[iter_] = ploss
                torch.save(pdict, os.path.join(self.path,'%s_best_%d.pt'%(self.prex,iter_)))
                res += 'save %s_best_%d.pt\t'%(self.prex,iter_)
        if not flag:
            torch.save(pdict, os.path.join(self.path,'%s_latest.pt'%self.prex))  
            res = 'update %s_latest.pt'%(self.prex)
        return res  
   
