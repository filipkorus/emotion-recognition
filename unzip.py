import zipfile

def unzip(dataset):
    file_name = f'{dataset}.zip'
    print(f'Unzipping {file_name}')

    with zipfile.ZipFile(f'data/{file_name}', 'r') as z:
        z.extractall('data')

    print(f'{dataset} unzipped')
