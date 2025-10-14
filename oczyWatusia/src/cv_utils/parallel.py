import multiprocessing
import time

class Process(multiprocessing.Process):
    def __init__(self, id):
        super(Process, self).__init__()
        self.id = id

    def run(self, func = None, args = None, kwargs = None):
        func(*args, **kwargs)