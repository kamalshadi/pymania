def is_loaded(func):
    def wrapper(obj):
        assert obj._loaded, "Data is not loaded!"
        return func(obj)
    return wrapper

class pipeline:
    def __init__(self, order):
        self.order = order

    def __call__(self, f):
        def wrapper(obj):
            if self.order>=0:
                assert self.order<=obj.order+1, "Out of order pipeline run"
            else:
                assert self.order<=obj.num_steps, "Out of order pipeline run"
            f(obj)
            obj.order = self.order
        return wrapper
