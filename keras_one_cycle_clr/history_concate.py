class History(object):
    
    def __init__(self, history=None):
        if history is not None:
            self.epoch = history.epoch
            self.history = history.history
        else:
            self.epoch = []
            self.history = {}
            

def concatenate_history(hlist, reindex_epoch=False):
    
    his = History()
    
    for h in hlist:
        his.epoch = his.epoch + h.epoch
        
        for key, value in h.history.items():
            his.history.setdefault(key, [])
            his.history[key] = his.history[key] + value
            
    if reindex_epoch:
        his.epoch = list(np.arange(0, len(his.epoch)))
        
    return his