class Result:
    def __init__(
        self               ,
        type        : str  ,
        loss        : float,
        time_elapsed: float):
        self.type         = type
        self.loss         = loss
        self.time_elapsed = time_elapsed

    def __str__(self):
        return f'{ self.type } | Loss: { self.loss } | Time Elapsed: { self.time_elapsed }'

    def to_dict(self) -> dict:
        return {
            'type'        : self.type,
            'loss'        : self.loss,
            'time_elapsed': self.time_elapsed
        }

    @staticmethod
    def from_dict(data: dict) -> 'Result':
        return Result(
            type         = data[ 'type'         ],
            loss         = data[ 'loss'         ],
            time_elapsed = data[ 'time_elapsed' ])
