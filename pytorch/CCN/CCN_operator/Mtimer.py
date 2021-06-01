import torch
class Timer():

    def __init__(self, flag=False):
        self.start_t = torch.cuda.Event(enable_timing=True)
        self.end_t = torch.cuda.Event(enable_timing=True)
        self.flag = flag

    def start(self):
        if self.flag:
            self.start_t.record()

    def end(self, out_string=''):
        if self.flag:
            self.end_t.record()
            torch.cuda.synchronize()
            print(out_string, self.start_t.elapsed_time(self.end_t))