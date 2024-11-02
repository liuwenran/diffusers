import torch
import os
from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
# base_prompt = ', full body, standing, A pose, white background, cartoon character, perfect, characters in the animation, highly accurate representations, and full of life.'
base_prompt = ", full body, standing, A pose, Facing forward, characters in the animation, white background, clear, cartoon character, highly accurate representations, even lighting with no shadows, No shadow between the feet"
# Depending on the variant being used, the pipeline call will slightly vary.
# Refer to the pipeline documentation for more details.

save_dir = 'results/flux_human_whitebg'
character_names = [
'Naruto Uzumaki',
'Sasuke Uchiha',
'Sakura Haruno',
'Kakashi Hatake',
'Goku',
'Vegeta',
'Bulma',
'Gohan',
'Luffy',
'Zoro',
'Nami',
'Sanji',
'Usopp',
'Ichigo Kurosaki',
'Rukia Kuchiki',
'Aizen',
'Light Yagami',
'L',
'Mikasa Ackerman',
'Eren Yeager',
'Levi Ackerman',
'Armin Arlert',
'Edward Elric',
'Alphonse Elric',
'Roy Mustang',
'Winry Rockbell',
'Spike Spiegel',
'Faye Valentine',
'Vash the Stampede',
'Meryl Stryfe',
'Ash Ketchum',
'Misty',
'Brock',
'Sailor Moon',
'Sailor Mercury',
'Sailor Mars',
'Sailor Jupiter',
'Sailor Venus',
'Inuyasha',
'Kagome Higurashi',
'Sesshomaru',
'Miroku',
'Ranma Saotome',
'Akane Tendo',
'Ryoga Hibiki',
'Shampoo',
'Yusuke Urameshi',
'Kurama',
'Hiei',
'Kuwabara',
'Kenshin Himura',
'Kaoru Kamiya',
'Saito Hajime',
'Shishio Makoto',
'Gon Freecss',
'Killua Zoldyck',
'Kurapika',
'Leorio',
'Natsu Dragneel',
'Lucy Heartfilia',
'Gray Fullbuster',
'Erza Scarlet',
'Meliodas',
'Elizabeth Liones',
'Ban',
'King',
'Kirito',
'Asuna',
'Sinon',
'Klein',
'Kaneki Ken',
'Touka Kirishima',
'Arima Kishou',
'Hinata Hyuga',
'Gaara',
'Jiraiya',
'Shikamaru Nara',
'Neji Hyuga',
'Rock Lee',
'Tenten',
'Temari',
'Kankuro',
'Orochimaru',
'Itachi Uchiha',
'Madara Uchiha',
'Obito Uchiha',
'Tsunade',
'Minato Namikaze',
'Kushina Uzumaki',
'Haku',
'Zabuza Momochi',
'Kiba Inuzuka',
'Shino Aburame',
'Choji Akimichi',
'Ino Yamanaka',
'Kuwabara',
'Faye Valentine',
'Vash the Stampede',
'Ryoga Hibiki',
'Shampoo',
]

for ind, character in enumerate(character_names):
    # if ind < 39:
    #     continue
    character_save_path = character.replace(' ', '_')
    folder_path = os.path.join(save_dir, character_save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        continue

    prompt = f"{character}{base_prompt}"
    for i in range(10):
        image = pipe(prompt, num_inference_steps=40).images[0]
        save_name = os.path.join(folder_path, f'{i}.png')
        image.save(save_name)