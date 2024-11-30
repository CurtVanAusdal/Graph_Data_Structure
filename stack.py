
class Stack:
    def __init__(self, size=0, items=[]):
        self.stack_size = size
        self.items = items

    def push(self, item):
        self.items.append(item)
        self.stack_size = len(self.items)

    def pop(self):

        beforepopped = self.items[-1]

        if self.stack_size == 0 or self.size == None:
            raise IndexError('the size of this stack is zero ')
        else:
            self.items.pop()
            self.stack_size = len(self.items)

        return beforepopped

    def top(self):
        if self.stack_size == 0:
            raise IndexError('there is no top because the size of the stack is zero ')
        else:
            return self.items[-1]

    def is_empty(self):
        if self.size() ==0:
            return True
        else:
            return False

    def size(self):
        self.stack_size = len(self.items)
        return (self.stack_size)

    def clear(self):
        self.items = []
        self.stack_size = len(self.items)
