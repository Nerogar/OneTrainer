from enum import Enum


class TimeUnit(Enum):
    EPOCH = 'EPOCH'
    STEP = 'STEP'
    SECOND = 'SECOND'
    MINUTE = 'MINUTE'
    HOUR = 'HOUR'

    NEVER = 'NEVER'
    ALWAYS = 'ALWAYS'

    def __str__(self):
        return self.value

    def is_time_unit(self) -> bool:
        return self == TimeUnit.SECOND \
            or self == TimeUnit.MINUTE \
            or self == TimeUnit.HOUR