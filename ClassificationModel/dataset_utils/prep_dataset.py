import os
import pathlib
from pathlib import Path
import shutil


# from tensorflow.python.lib.io.file_io import create_dir


def rename_all_files_in_directory(directory_path: str, name_pattern: str = None):
    pathlib.Path(directory_path)
    directory = os.listdir(path=directory_path)
    i = 0
    for file in directory:
        os.rename(rf'{directory_path}/{file}', rf'{directory_path}/{name_pattern}{i}')
        i += 1
    print('success!')


def copy_files(src, dest, name, start, stop):
    fail_names = [f'{name}{i}.png' for i in range(start, stop)]
    for fail_name in fail_names:
        src_file = src / fail_name
        dest_file = dest / fail_name
        shutil.copy(src_file, dest_file)


def create_dir(dir_name, children):
    new_dir = dir_name / children
    new_dir.mkdir()
    return new_dir


def main():
    print('prep_dataset start')
    base_dir = create_dir(Path('.'), 'dataset')
    train_dir = create_dir(base_dir, 'train')
    test_dir = create_dir(base_dir, 'test')
    valid_dir = create_dir(base_dir, 'validation')

    train_ben_dir = create_dir(train_dir, 'ben')
    train_mal_dir = create_dir(train_dir, 'mal')
    test_ben_dir = create_dir(test_dir, 'ben')
    test_mal_dir = create_dir(test_dir, 'mal')
    valid_ben_dir = create_dir(valid_dir, 'ben')
    valid_mal_dir = create_dir(valid_dir, 'mal')

    src = Path('.') / 'tmp' / 'mal_and_ben_images'
    copy_files(src, train_ben_dir, 'ben', 0, 321)
    copy_files(src, train_mal_dir, 'mal', 0, 321)
    copy_files(src, test_ben_dir, 'ben', 321, 374)
    copy_files(src, test_mal_dir, 'mal', 321, 374)
    copy_files(src, valid_ben_dir, 'ben', 374, 427)
    copy_files(src, valid_mal_dir, 'mal', 374, 427)

    print('prep_dataset success')


if __name__ == '__main__':
    main()
