"""Expected geometry/input failures with actionable machine-readable evidence."""


class GeometryInputError(ValueError):
    def __init__(self,code,message,**details):
        self.code=code;self.details=details
        super().__init__(f'{code}: {message}')
