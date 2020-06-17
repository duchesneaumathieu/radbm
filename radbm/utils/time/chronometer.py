from time import time

class Chronometer(object):
    """
    A chronometer found time codes.

    Methods
    -------
    reset()
        Resets the chronometer.
    start()
        Starts the chronometer.
    stop()
        Stops the chronometer.
    time() : float
        return the number of second on the chronometer.

    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.running = False
        self.counter = 0
        return self
        
    def start(self):
        if self.running:
            raise ValueError('cannot start if running')
        self.running = True
        self.t0 = time()
        return self
        
    def stop(self):
        if not self.running:
            raise ValueError('cannot stop if not running')
        self.counter += time() - self.t0
        self.running = False
        return self
    
    def time(self):
        if self.running:
            return self.counter + time() - self.t0
        return self.counter
    
    def __repr__(self):
        state = 'start' if self.running else 'stop'
        return '{:.4f} ({})'.format(self.time(), state)