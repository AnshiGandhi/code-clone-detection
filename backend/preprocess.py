import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_files(directories):
    """Returns a list of all files from multiple directories."""
    files = []
    for directory in directories:
        files.extend(Path(directory).rglob('*'))
    return files

def process_file(item, index):
    """Processes a single file and returns its JSON entry."""
    try:
        return {
            'label': item.parts[-2],  # Folder name as label
            'index': str(index),
            'code': item.read_text(encoding='latin-1')
        }
    except Exception as e:
        print(f"Error reading {item}: {e}")
        return None

def process_files_in_parallel(files):
    """Processes files in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor() as executor:
        # Map each file with its corresponding index
        data = list(tqdm(
            executor.map(lambda p: process_file(p[0], p[1]), zip(files, range(len(files)))), 
            total=len(files)
        ))
    return [d for d in data if d is not None]  # Filter out any failed reads

def write_jsonl(file_path, data, batch_size=1000):
    """Writes the given data to a JSONL file in batches."""
    with open(file_path, 'w') as f:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            f.writelines(json.dumps(entry) + '\n' for entry in batch)

# Dataset directories
train_dirs = [f"ProgramData/{i}" for i in range(1, 65)]
valid_dirs = [f"ProgramData/{i}" for i in range(65, 81)]
test_dirs = [f"ProgramData/{i}" for i in range(81, 105)]

# Process and write datasets
train_files = get_files(train_dirs)
train_data = process_files_in_parallel(train_files)
write_jsonl("train.jsonl", train_data)

valid_files = get_files(valid_dirs)
valid_data = process_files_in_parallel(valid_files)
write_jsonl("valid.jsonl", valid_data)

test_files = get_files(test_dirs)
test_data = process_files_in_parallel(test_files)
write_jsonl("test.jsonl", test_data)