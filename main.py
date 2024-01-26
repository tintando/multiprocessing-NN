import os

def empty_files(directory):
    print('Emptying files in directory: ' + directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file != 'main.py':
                file_path = os.path.join(root, file)
                print(file_path)
                with open(file_path, 'w') as f:
                    f.write('')

# Specify the directory where the file is located
directory = 'C:/Users/tintando/Desktop/ParallelPCA-main/ParallelPCA-main'

# Call the function to empty the files
empty_files(directory)
