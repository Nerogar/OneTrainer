from enum import Enum


class CloudFileSync(Enum):
    FABRIC_SFTP = 'FABRIC_SFTP'
    NATIVE_SCP = 'NATIVE_SCP'
    def __str__(self):
        return self.value
