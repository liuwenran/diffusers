import os
from PIL import Image

image = Image.open('/mnt/petrelfs/liuwenran/forks/diffusers/results/test_liuwenran/0.png')

folder_path = 'results/test_save'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(100):
    print('save ' + str(ind))
    image.save(os.path.join(folder_path, str(ind) + ".png"))
