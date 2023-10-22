

import os


full_path_to_images = \
    'C:\\Users\\123\\OIDv4_ToolKit\\OID\\Dataset\\train\\Car_Person_Dog_Horse'

print(os.getcwd())

os.chdir(full_path_to_images)

print(os.getcwd())

# se defineste o lista pentru a scrie caile in ea
p = []

# Folosind os.walk parcurgem toate fisierele
# in curentul director
# Fullstop in os.walk('.') means the current directory
for current_dir, dirs, files in os.walk('.'):
    # mergem prin toate fisierele
    for f in files:
        # numele fisierului sa se termine in '.jpg'
        if f.endswith('.jpg'):
            # pregatirea calei train.txt file
            path_to_save_into_txt_files = full_path_to_images + '/' + f

            #trecerea la un nou rand in txt
            p.append(path_to_save_into_txt_files + '\n')


# Impartim primile 15% elemente din lista
# pentru a le scrie in test.txt
p_test = p[:int(len(p) * 0.15)]

# stergem din lista initiala primele 15% elemente
p = p[int(len(p) * 0.15):]


# Cream train.txt si scriem 85% din randuri in fisier
with open('train.txt', 'w') as train_txt:
    # parcurgem toatele elementele listei
    for e in p:
        # Scriem calea curenta la finalul fisierului
        train_txt.write(e)

# Cream train.txt si scriem 15% din randuri in fisier
with open('test.txt', 'w') as test_txt:
    # parcurgem toatele elementele listei
    for e in p_test:
        # Scriem calea curenta la finalul fisierului
        test_txt.write(e)

