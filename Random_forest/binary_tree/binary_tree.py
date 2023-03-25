class Node:
    def __init__(self, data) -> None:
        self.left=None
        self.right=None 
        self.data=data

    def insert(self, data, condition: bool):
        if self.data:
            if condition:
                if self.left is None:
                    self.left=Node(data)
                else:
                    self.left.insert(data, condition)
            else:
                if self.right is None:
                    self.right=Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data=data

    
