#User function Template for python3

def isQueueSymmetric(queue):
    
    for i in range(0, int(len(queue)/2)):
        if queue[i].val != queue[-(i + 1)].val:
            return False
    return True
    
def isSymmetric(root):
    queue = list()
    queue.insert(0, root)
    while queue:
        children = list()
        while queue:
            child = queue.pop()
            if child.left:
                children.append(child.left)
            if child.right:
                children.append(child.right)
        
        if not isQueueSymmetric(children):
            return False
        
        queue = children
    
    return True    

class Node:
    def __init__(self, key):
        self.right = None
        self.left = None
        self.val = key
