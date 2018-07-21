#!../bin/python3.6

class VReaderBaseError(Exception):
    pass

class VReaderBase:
    def read(self, logic_time):
        raise VReaderBaseError("you must implement your own read function")

class VReaderSteady(VReaderBase):
    def read(self,logic_time):
        return 0