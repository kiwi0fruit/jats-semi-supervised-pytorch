import os
from os.path import dirname, join, abspath, isfile
import subprocess
from pathlib import Path

# Mode #1: Simply run but make sure that ../train_log is empty
# Mode #2: Simply run but make sure that ../train_log has logs
#   with successful or good enough to try runs copied from
#   ../train_log_search_model . Then set OFFSET_EP below to
#   the last epoch from latest checkpoint + 1.
#   (Placing "skip" file into the directory would skip it).

MAX_ERROR = 1000

traindir = dirname(abspath(__file__))
logdir = join(dirname(traindir), 'train_log')
Path(logdir).mkdir(parents=True, exist_ok=True)
versiondirs_in_logdir = sorted(os.listdir(logdir))


def run_until_found_1():
    for i in range(MAX_ERROR):
        # noinspection PyBroadException
        try:
            ret = subprocess.run(['python', join(traindir, 'training.py'),
                                  '--name', 'train_log_search_model'])
        except Exception as ex:
            print(ex)
            continue
        if ret.returncode != 0:
            continue
        break


def re_run_until_found_2():
    for verdir in versiondirs_in_logdir:
        skipfile = join(logdir, verdir, 'skip')
        if isfile(skipfile):
            continue

        # noinspection PyBroadException
        try:
            ret = subprocess.run(['python', join(traindir, 'training.py'), '--ver', verdir])
        except Exception as ex:
            print(ex)
            print('', file=open(skipfile, 'w'))
            continue
        if ret.returncode != 0:
            print('', file=open(skipfile, 'w'))
            continue
        break
    else:
        run_until_found_1()


if __name__ == '__main__':
    re_run_until_found_2()
