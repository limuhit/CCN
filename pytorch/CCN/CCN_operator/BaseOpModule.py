import torch
from torch import nn
from collections import OrderedDict

class BaseOpModule(nn.Module):

    def __init__(self, devices=0):
        super(BaseOpModule,self).__init__()
        self.device_list = [devices] if isinstance(devices,int) else devices
        self.apply_flag = False

    def _apply(self,fn):
        super(BaseOpModule,self)._apply(fn)
        self.apply_flag = True
        fn(self)
        return self
    
    def custom_op_replicate(self,other):
        other.op = self.op
        return other

    def _replicate_for_data_parallel(self):
        replica = self.__new__(type(self))
        #replica.op = self.op
        replica = self.custom_op_replicate(replica)
        replica.__dict__ = self.__dict__.copy()
        replica._parameters = OrderedDict()
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()
        replica._is_replica = True
        return replica

    def custom_op_to(self, *args):
        if args[0] is not None and len(self.op.keys())==1:   
            new_id = args[0].index
            old_id = list(self.op.keys())[0]
            if not (new_id == old_id): 
                self.op[new_id] = self.op.pop(old_id)
                self.op[new_id].to(new_id)
                

    def is_floating_point(self):
        return False
    
    def is_complex(self):
        return False

    def to(self, *args, **kwargs):
        #print("dtow to", args)
        if not self.apply_flag:
            super(BaseOpModule,self).to(*args, **kwargs)
        else:
            self.custom_op_to(*args)
            self.apply_flag = False
        return self