from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.is_train = False
        parser = self._parser
