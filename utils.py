import os, shutil 
import glob 
import random

def move_random_files(from_dir, to_dir, file_format, perc=0.1):
    '''
    Function for moving a random percentage of files (with a specified extension)
    between to directories.
    '''

    assert os.path.isdir(from_dir), 'Source directory does not exist!'
    
    if not os.path.isdir(to_dir):
        os.makedirs(to_dir)
    
    image_list = glob.glob(f'{from_dir}/*.{file_format}')
    print(f'There are {len(image_list)} .{file_format} files in the source directory.')
    n_to_move = int(perc * len(image_list))
    print(f'Moving {n_to_move} files...')
    files_to_move = random.sample(image_list, n_to_move)
    
    for file in files_to_move:
        
        f = os.path.basename(file)
        src = os.path.join(from_dir, f)
        dst = os.path.join(to_dir, f)
        shutil.move(src, dst)
    
    print(f'done.')
    
def get_random_file(from_dir, file_format):
    
    assert os.path.isdir(from_dir), 'Source directory does not exist!'
    
    file_list = glob.glob(f'{from_dir}/*.{file_format}')
    return random.sample(file_list, 1)[0]
    