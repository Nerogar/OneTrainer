from modules.util.enum.BaseEnum import BaseEnum


class TimeUnit(BaseEnum):
    EPOCH = 'EPOCH'
    STEP = 'STEP'
    SECOND = 'SECOND'
    MINUTE = 'MINUTE'
    HOUR = 'HOUR'

    NEVER = 'NEVER'
    ALWAYS = 'ALWAYS'

    def is_time_unit(self) -> bool:
        return self == TimeUnit.SECOND \
            or self == TimeUnit.MINUTE \
            or self == TimeUnit.HOUR
