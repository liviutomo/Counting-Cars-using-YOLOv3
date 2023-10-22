
full_path_to_images = \
    'C:\\Users\\123\\OIDv4_ToolKit\\OID\\Dataset\\train\\Car_Person_Dog_Horse'

# Definim counter pentru clase
c = 0

# Cream classes.names din existentul classes.txt
with open(full_path_to_images + '/' + 'classes.names', 'w') as names, \
     open(full_path_to_images + '/' + 'classes.txt', 'r') as txt:
    # parcurgem toate liniile in fisierul txt si le scriem in fisierul name
    for line in txt:
        names.write(line)
        c += 1
# Cream custom_data.data
with open(full_path_to_images + '/' + 'custom_data.data', 'w') as data:
    # Scriem cele 5 lini dorite
    # numarul de clase
    # Folosind '\n' mergem la urmatoarea linie
    data.write('classes = ' + str(c) + '\n')
    # Locatia train.txt
    data.write('train = ' + full_path_to_images + '/' + 'train.txt' + '\n')
    # Locatia test.txt
    data.write('valid = ' + full_path_to_images + '/' + 'test.txt' + '\n')
    # Locatia classes.names
    data.write('names = ' + full_path_to_images + '/' + 'classes.names' + '\n')
    # Locatia unde sa salvam ponderile rezultate
    # in urma antrenarii
    data.write('backup = backup')


