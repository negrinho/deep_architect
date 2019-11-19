import subprocess
import random
import sys
import os


def get_empty_bucket_folder(folder):
    folder_name = ''
    while True:
        num = random.randint(0, sys.maxsize)
        folder_name = os.path.join(folder, 'eval' + str(num))
        try:
            subprocess.check_call(['gsutil', '-q', 'stat', folder_name + '/**'])
        except subprocess.CalledProcessError:
            break
    return folder_name


def delete_bucket_folder(folder):
    try:
        subprocess.check_call(['gsutil', '-m', 'rm', folder + '/**'])
    except subprocess.CalledProcessError:
        pass


def get_gcloud_project():
    try:
        return subprocess.check_output(
            ['gcloud', 'config', 'get-value', 'project'])
    except subprocess.CalledProcessError:
        return ''


def get_gcloud_zone():
    try:
        return subprocess.check_output(
            ['gcloud', 'config', 'get-value', 'compute/zone'])
    except subprocess.CalledProcessError:
        return ''
