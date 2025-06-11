import multiprocessing

class Consumer(multiprocessing.Process):
    """ For parallelisation """
    
    def __init__(self,task_queue,task_function,locks=None,result_queue=None):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.locks = locks
        self.task_function = task_function
        self.result_queue = result_queue
        
    def run(self):
        proc_name = self.name
        while True:
            next_task_args = self.task_queue.get()
            if next_task_args is None:
                self.task_queue.task_done()
                break
            result = self.task_function(*next_task_args,self.locks)    
            self.task_queue.task_done()
            if self.result_queue is not None:
                self.result_queue.put(result)

def end_queue(task_queue,n_processes):
    for _ in range(n_processes):
        task_queue.put(None)
    return task_queue