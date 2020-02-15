from enum import Enum


class TokenType(Enum):
    NUM = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    LBRACE = 6
    RBRACE = 7
    INVALID = 8

    @staticmethod
    def type(x):
        if str(x).isnumeric():
            return TokenType.NUM

        if x == '+':
            return TokenType.ADD

        if x == '-':
            return TokenType.SUB

        if x == '*':
            return TokenType.MUL

        if x == '/':
            return TokenType.DIV

        if x == '(':
            return TokenType.LBRACE

        if x == ')':
            return TokenType.RBRACE

        return TokenType.INVALID

    @staticmethod
    def isbinoperator(x):
        return True if x == '+' or x == '-' or x == '*' or x == '/' else False

    @staticmethod
    def isoperater(x):
        return True if x == '+' or x == '-' or x == '*' or x == '/' else False

    @staticmethod
    def isoperator_add_sub(x):
        return True if x == '+' or x == '-' else False

    @staticmethod
    def isoperator_mul_div(x):
        return True if x == '*' or x == '/' else False


class Token:

    def __init__(self, val):
        self.val = str(val)
        self.type: TokenType = TokenType.type(self.val)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.val == other

        if isinstance(other, Token):
            return self.val == other.val

        return False

    def __gt__(self, other):
        if self.type == TokenType.NUM and other.type == TokenType.NUM:
            return True if self.val > other.val else False

        if TokenType.isoperater(self.val) and TokenType.isoperater(other.val):
            return True if TokenType.isoperator_mul_div(self.val) and \
                           TokenType.isoperator_add_sub(other.val) else False

    def __str__(self):
        return str(self.val)


class Stack:

    def __init__(self):
        self.stack = []

    def push(self, x):
        self.stack.append(x)

    def pop(self):
        return self.stack.pop()

    def is_empty(self):
        return False if self.stack else True

    def peek(self):
        return self.stack[-1] if not self.is_empty() else None

    def __str__(self):
        return str(self.stack)


def infix2postfix(infix: str):
    stack = Stack()
    post_fix: str = ''

    for ch in infix:
        token = Token(ch)

        if token.type == TokenType.NUM:
            post_fix += token.val

        if token == '(':
            stack.push(token)

        if token == ')':
            while stack.peek() != '(':
                post_fix += stack.pop().val
            stack.pop()

        if TokenType.isoperater(token):

            if not TokenType.isoperater(stack.peek()):
                stack.push(token)
                continue

            if token > stack.peek():
                stack.push(token)
            else:
                post_fix += stack.pop().val
                stack.push(token)

    while not stack.is_empty():
        post_fix += stack.pop().val

    return post_fix


def eval(postfix: str):
    stack = Stack()
    for x in postfix:
        token = Token(x)

        if TokenType.type(token.val) == TokenType.NUM:
            stack.push(token)

        if TokenType.isbinoperator(token.val):
            op2 = stack.pop().val
            op1 = stack.pop().val
            val = 0
            if token.val == '+':
                val = int(op1) + int(op2)

            if token.val == '-':
                val = int(op1) - int(op2)

            if token.val == '*':
                val = int(op1) * int(op2)

            if token.val == '/':
                val = int(op1) / int(op2)

            stack.push(Token(str(val)))

    return int(stack.pop().val)


if __name__ == "__main__":
    postfix = infix2postfix("(1 + 3*2 + 2*(3 - 4*5))")
    print(eval(postfix))
