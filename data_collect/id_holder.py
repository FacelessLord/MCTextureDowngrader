
class IdHolder:
    def __init__(self):
        self.__id = 0

    def next(self):
        old_id = self.__id
        self.__id = self.__id + 1

        return old_id