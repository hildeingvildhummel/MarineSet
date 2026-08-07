def sort_AISgekoppeld(excel_file):
    import pandas as pd
    info = pd.read_excel(excel_file)
    info['Audio Start'] = pd.to_datetime(info['Audio Start'])
    info['Audio End'] = pd.to_datetime(info['Audio End'])
    return (
        info.assign(tmp=info["Audio End"] - info["Audio Start"])
        .sort_values(by="tmp", ascending=False)
        .drop(columns="tmp")
    )

def clean_folder(folder):
    import os, shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
