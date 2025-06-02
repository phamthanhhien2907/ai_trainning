import string
import random
import os

current_dir = os.getcwd()
print("current_dir", current_dir)

def generateRandomString(str_length: int = 20):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(str_length))