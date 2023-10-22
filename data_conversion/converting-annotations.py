

import pandas as pd
import os

full_path_to_csv = 'C:\\Users\\123\\OIDv4_ToolKit\\OID\\csv_folder'

full_path_to_images = \
    'C:\\Users\\123\OIDv4_ToolKit\\OID\\Dataset\\train\\Car'

labels = ['Car']

classes = pd.read_csv(full_path_to_csv + '/' + 'class-descriptions-boxable.csv',
                      usecols=[0, 1], header=None)


encrypted_strings = []

# Preluam fiecare cod pentru fiecare clasa de interes
# parcurgem toate etichetele
for v in labels:
    # Getting Pandas sub-dataFrame that has only one row
    # Prin 'loc' localizam randul care satisface
    #  conditia 'classes[1] == v'
    sub_classes = classes.loc[classes[1] == v]

    # Luam elementul de pe prima linie, prima coloana
    e = sub_classes.iloc[0][0]
    #print(e)  # /m/0k4j

    # se adauga in lista
    encrypted_strings.append(e)

# citim fisierul csv cu annotations

annotations = pd.read_csv(full_path_to_csv + '/' + 'train-annotations-bbox.csv',
                          usecols=['ImageID',
                                   'LabelName',
                                   'XMin',
                                   'XMax',
                                   'YMin',
                                   'YMax'])


# Localizam doar codarile necesare
# si le copiem in sub_ann
sub_ann = annotations.loc[annotations['LabelName'].isin(encrypted_strings)].copy()


sub_ann['classNumber'] = ''
sub_ann['center x'] = ''
sub_ann['center y'] = ''
sub_ann['width'] = ''
sub_ann['height'] = ''

# Parcurgem toate clasele codate
# si le convertim in numere
for i in range(len(encrypted_strings)):
    # scriem numerele in coloana potrivita
    sub_ann.loc[sub_ann['LabelName'] == encrypted_strings[i], 'classNumber'] = i

# calculam centrul casetei pentru x si y
sub_ann['center x'] = (sub_ann['XMax'] + sub_ann['XMin']) / 2
sub_ann['center y'] = (sub_ann['YMax'] + sub_ann['YMin']) / 2

# calculam intaltimea si latimea bounding boxului
sub_ann['width'] = sub_ann['XMax'] - sub_ann['XMin']
sub_ann['height'] = sub_ann['YMax'] - sub_ann['YMin']

# elementele din fisierul csv (image id, clasa + coordonate
# vor fi puse in fisiere txt
r = sub_ann.loc[:, ['ImageID',
                    'classNumber',
                    'center x',
                    'center y',
                    'width',
                    'height']].copy()

"""
Salvam notatiile in fisier txt
"""

# afisarea locatiei curente
print(os.getcwd())

# schimbam directorul curent cu cel al imaginilor
os.chdir(full_path_to_images)

# Check point

print(os.getcwd())

# parcurgem toate fisierele din directorul curent
for current_dir, dirs, files in os.walk('.'):

    for f in files:
        # verificam daca sunt imagini '.jpg'
        if f.endswith('.jpg'):
            # Slicing only name of the file without extension
            image_name = f[:-4]
            # se creaza fisere txt cu acelasi nume ca si al pozelor
            sub_r = r.loc[r['ImageID'] == image_name]
                    # se noteaza in txt clasa, si dimensiunile casetei
            resulted_frame = sub_r.loc[:, ['classNumber',
                                           'center x',
                                           'center y',
                                           'width',
                                           'height']].copy()

            # Pregatim o cale un sa salvam fisierul txt

            path_to_save = full_path_to_images + '\\' + image_name + '.txt'

            # salvam datele in fisier txt
            resulted_frame.to_csv(path_to_save, header=False, index=False, sep=' ')


