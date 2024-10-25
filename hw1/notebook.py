from threading import Thread, Lock
import time, random
import numpy as np

class MyThread(Thread):
    def __init__(self, index):
        Thread.__init__(self)

        self.index = index

    def run(self):
        time.sleep(3)
        print('Hello, i am thread number' + str(self.index))


thread_list = []
for i in range(5):
    thread_list.append(MyThread(i))

for i in range(5):
    thread_list[i].start()


class Fork():
    def __init__(self):
        self.mutex = Lock()

    def acquire(self):
        self.mutex.acquire()

    def put_down(self):
        self.mutex.release()


fork = Fork()

class DumbPhilosopher(Thread):
    def __init__(self, name, fork_left, fork_right):
        Thread.__init__(self)
        self.name = name
        self.fork_left = fork_left
        self.fork_right = fork_right

    def eat(self):
        print("Philosopher " + str(self.name) + " start eating.")
        time.sleep(random.uniform(0, 5))

    def think(self):
        print("Philosopher " + str(self.name) + " think a lot.")
        time.sleep(random.uniform(0, 5))

    def take_fork(self):
        print("Philosopher " + str(self.name) + " try to catch forks.")
        self.fork_left.acquire()
        self.fork_right.acquire()
        
        print("Philosopher " + str(self.name) + " has caught forks.")

    def put_forks_down(self):
        print("Philosopher " + str(self.name) + " has eaten a lot and put down forks.")
        self.fork_left.put_down()
        self.fork_right.put_down()

    def run(self):
        while (True):
            self.think()
            self.take_fork()
            self.eat()
            self.put_forks_down()


names = ['Аристотель', 'Конфуций', 'Фрейд', 'Декарт', 'Ницше']

forks = []
for i in range(len(names)):
    fork.append(Fork())

philosophers = []
for i in range(len(names)):
    philosophers.append(DumbPhilosopher(names[i], forks[i - 1], forks[i]))

for i in range(len(names)):
    philosophers[i].start()