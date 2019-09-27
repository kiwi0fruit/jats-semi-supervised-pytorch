import os
from os import path as p
from kiwi_bugfix_typechecker import ipython


class Log:
    def __init__(self, temp_dir: str, run_id: str, db_name: str):
        self.db_name = db_name
        self.temp_dir = temp_dir
        self.run_id = run_id
        os.makedirs(p.dirname(self.checkpoint()), exist_ok=True)
        self.set_run_id(run_id)

    def set_run_id(self, run_id: str):
        self.run_id = run_id

        string = self.prefix_nn() + '.txt'
        if not p.exists(string):
            print('', file=open(string, 'w'))
        string = self.prefix_nn() + '_i.txt'
        if not p.exists(string):
            print('', file=open(string, 'w'))

    def prefix_db(self) -> str:
        return p.join(self.temp_dir, f'jats-db-{self.db_name}')

    def prefix_nn(self) -> str:
        return self.prefix_db() + (f'__id-{self.run_id}' if self.run_id else '')

    def checkpoint(self, epoch: int=None) -> str:
        return self.prefix_nn() + (f'__ep{epoch}' if epoch else '') + '.pt'

    def print(self, *objs):
        print(*objs)
        print(*objs, file=open(self.prefix_nn() + '.txt', 'a', encoding='utf-8'))

    def print_i(self, *objs):
        print(*objs, file=open(self.prefix_nn() + '_i.txt', 'a', encoding='utf-8'))

    def display(self, *objs):
        ipython.display(*objs)
        print(*objs, file=open(self.prefix_nn() + '.txt', 'a', encoding='utf-8'))
