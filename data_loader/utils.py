import os
import gdown
import zipfile


def download_and_extract_from_google_drive(url, destination_path):
    """
    download the file from url and extract the file into destination_path
    :param url:
    :param destination_path:
    :return:
    """
    if not os.path.exists(destination_path):
        os.makedirs(destination_path, exist_ok=True)
    temp_file_path = os.path.join(destination_path, 'temp.zip')

    gdown.download(url, temp_file_path, quiet=False)
    with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)
    # os.remove(temp_file_path)


if __name__ == '__main__':
    from data_loader.data_registry import UET_ECHO
    download_and_extract_from_google_drive(UET_ECHO, 'data/uet_echo')
    print('Done!')

