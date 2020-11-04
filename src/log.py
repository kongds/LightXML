import datetime

class Logger:
    def __init__(self, name):
        self.name = name

    def log(self, text):
        with open(f'./log/{self.name}', 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')
