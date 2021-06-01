import os

class Logger():
    
    def __init__(self, fname, screen = True, file = True):
        self.file = file
        self.fout = open(fname, 'w') if file else None
        self.screen_out = screen
    
    def log(self, *args):
        if self.screen_out:
            print(*args)
        if self.file:
            self.fout.write(*args)
            self.fout.write('\n')
            self.fout.flush()
    
    def close(self):
        if self.file: self.fout.close()

    def __del__(self):
        print("close")
        self.close()

if __name__ == '__main__':
    log = Logger('test_log.txt')
    s=1
    log.log("this is a test {:d}".format(s))
    log.log("hello")
