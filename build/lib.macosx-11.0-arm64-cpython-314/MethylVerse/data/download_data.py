from os.path import join
import os
import requests
import shutil
from tqdm.auto import tqdm
import json
import gzip
import glob
import zipfile


def gunzip_file(input_path, output_path):
    """
    Decompresses a .gz file.

    Args:
        input_path (str): Path to the input .gz file.
        output_path (str): Path to save the decompressed output.
    """
    try:
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
    except FileNotFoundError:
         print(f"Error: Input file '{input_path}' not found.")
    except gzip.BadGzipFile:
        print(f"Error: '{input_path}' is not a valid gzip file.")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_file(file_dict):
    # Determine the path to the data directory
    destdir = os.path.split(os.path.realpath(__file__))[0]
    
    urlpath = file_dict["links"]["self"]
    name = file_dict["key"]
    total_length = file_dict["size"]
    
    # make an HTTP request within a context manager
    r = requests.get(urlpath, stream=True)
        
    # implement progress bar via tqdm
    with tqdm.wrapattr(r.raw, "read", total=total_length, desc="")as raw:
    
        # save the output to a file
        with open(join(destdir, name+".gz"), 'wb') as output:
            shutil.copyfileobj(raw, output)
    
    # Determine if gzipped
    encoded = "content-encoding" in list(r.headers.keys())
    r.close()

    # Unzip the file
    with zipfile.ZipFile(join(destdir, name+".gz"), 'r') as zip_ref:
        zip_ref.extractall(destdir)
    os.remove(join(destdir, name+".gz"))

    # Gunzip file
    #if encoded:
    #    gunzip_file(join(destdir, name+".gz"), join(destdir, name))
    #    os.remove(join(destdir, name+".gz"))
    #else:
    #    os.rename(join(destdir, name+".gz"), join(destdir, name))

    return None


def check_download():
    current_dir = os.path.split(os.path.realpath(__file__))[0]
    files = glob.glob(os.path.join(os.path.join(current_dir,"**"),"*.parquet"), recursive=True)
    if len(files) == 0:
        return False
    else:
        return True
    

def check_MPACT_download():
    current_dir = os.path.split(os.path.realpath(__file__))[0]
    files = glob.glob(os.path.join(os.path.join(current_dir,"**"),"MPACT*.pth"), recursive=True)
    if len(files) == 0:
        return False
    else:
        return True
        

def download_methyl_anno():

    is_downloaded = check_download()
    if is_downloaded:
        return

    recordID = "16580408"
    url = 'https://zenodo.org/api/records/'
    r = requests.get(url+recordID)
    js = json.loads(r.text)

    for file_dict in js["files"]:
        download_file(file_dict)


def download_MPACT():

    is_downloaded = check_MPACT_download()
    if is_downloaded:
        return

    recordID = "16581863"
    url = 'https://zenodo.org/api/records/'
    r = requests.get(url+recordID)
    js = json.loads(r.text)

    for file_dict in js["files"]:
        download_file(file_dict)