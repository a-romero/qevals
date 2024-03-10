import os
import bz2
import requests
import zipfile
import tarfile
import datetime

def process_paths(path_array):
    result_paths = []

    def download_and_extract_http_file(http_url, target_dir):
        local_filename = http_url.split('/')[-1]
        local_path = os.path.join(target_dir, local_filename)
        
        # Download the file from the HTTP URL
        with requests.get(http_url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Decompress if necessary and remove the compressed file
        if zipfile.is_zipfile(local_path):
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            os.remove(local_path)
        elif tarfile.is_tarfile(local_path):
            with tarfile.open(local_path, 'r:*') as tar_ref:
                tar_ref.extractall(target_dir)
            os.remove(local_path)
        elif local_filename.endswith('.bz2'):
            with bz2.BZ2File(local_path, 'rb') as file:
                with open(local_path[:-4], 'wb') as new_file:
                    for data in iter(lambda: file.read(100 * 1024), b''):
                        new_file.write(data)
            os.remove(local_path)

    def process_path(path):
        if os.path.isfile(path):
            if path.endswith('.zip') or path.endswith('.tar') or path.endswith('.tar.gz') or path.endswith('.tgz'):
                target_dir = os.path.splitext(path)[0]
                os.makedirs(target_dir, exist_ok=True)
                if path.endswith('.zip'):
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        zip_ref.extractall(target_dir)
                else:
                    with tarfile.open(path, 'r:*') as tar_ref:
                        tar_ref.extractall(target_dir)
                result_paths.extend(process_paths(target_dir))
            else:
                result_paths.append(path)
        elif os.path.isdir(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                result_paths.extend(process_path(file_path))
        elif path.startswith('http://') or path.startswith('https://'):
            target_dir = "./data_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(target_dir, exist_ok=True)
            download_and_extract_http_file(path, target_dir)
            result_paths.extend(process_paths(target_dir))
        else:
            print(f"Path {path} is not supported")

    for path in path_array:
        process_path(path)

    return result_paths

def classify_files(path_array):
    text_files = []
    pdf_files = []
    docx_files = []

    for path in path_array:
        if path.endswith('.txt'):
            text_files.append(path)
        elif path.endswith('.pdf'):
            pdf_files.append(path)
        elif path.endswith('.docx') or path.endswith('.doc'):
            docx_files.append(path)

    return text_files, pdf_files, docx_files
