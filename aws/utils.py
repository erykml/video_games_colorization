# libraries 
import os, shutil 
import glob 
import random

def move_random_files(from_dir, to_dir, file_format, perc=0.1, seed=42):
    '''
    Function for moving a percentage of files (at random, with a specified extension)
    from one directory to another.
    
    Parameters
    ----------
    from_dir : str
        Path of the directory from which the files will be moved
    to_dir : str
        Path of the directory to which the files will be moved
    file_format : str
        String indicating the extension of the files to be moved, for example, `jpg`
    perc : numeric
        Fraction indicating the percentage of files to move, range (0, 1], default 10%
    seed : int
        Random seed for reproducibility
    '''

    assert os.path.isdir(from_dir), 'Source directory does not exist!'
    assert ((perc > 0) and (perc <= 1)), 'Invalid percentage!'
    
    if not os.path.isdir(to_dir):
        os.makedirs(to_dir)
    
    image_list = glob.glob(f'{from_dir}/*.{file_format}')
    print(f'There are {len(image_list)} .{file_format} files in the source directory.')
    n_to_move = int(perc * len(image_list))
    print(f'Moving {n_to_move} files...')
    random.seed(seed)
    files_to_move = random.sample(image_list, n_to_move)
    
    for file in files_to_move:
        
        f = os.path.basename(file)
        src = os.path.join(from_dir, f)
        dst = os.path.join(to_dir, f)
        shutil.move(src, dst)
    
    print(f'done.')
    
def get_random_file(from_dir, file_format):
    '''
    Function for retrieving a path to a random file with a specified 
    source directory and file extension.

    Parameters
    ----------
    from_dir : str
        Path of the directory from which the files will be moved
    file_format : str
        String indicating the extension of the files to be moved, for example, `jpg`

    Returns
    -------
    path : str
        Path to the random file
    '''
    
    assert os.path.isdir(from_dir), 'Source directory does not exist!'
    
    file_list = glob.glob(f'{from_dir}/*.{file_format}')
    return random.choice(file_list)
    
