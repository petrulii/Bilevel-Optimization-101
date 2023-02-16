
class Function:
    """
    A class that describes a function.
    """
    def __init__(self, f, grad=None, hess=None, name=None):
        self.f = f
        self.grad = grad
        self.hess = hess
        self.name = name