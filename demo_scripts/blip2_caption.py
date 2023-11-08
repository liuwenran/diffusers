# import torch
# from PIL import Image
# from lavis.models import load_model_and_preprocess

# # setup device to use
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# # load sample image
# # loads BLIP-2 pre-trained model
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)

# # prepare the image
# image_file = '/mnt/petrelfs/liuwenran/datasets/changshiban/changshiban_ancient_frames.txt'
# image_lines = open(image_file, 'r').readlines()

# result_file = '/mnt/petrelfs/liuwenran/datasets/changshiban/changshiban_ancient_frames_caption.txt'
# result_file = open(result_file, 'w')

# for ind, line in enumerate(image_lines):
#     print(f'image ind {ind}')
#     line = line.strip()
#     raw_image = Image.open(line).convert("RGB")

#     image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
#     answer = model.generate({"image": image, "prompt": "Question: what is the content in this picture, in detail? Answer:"})
#     answer = answer[0]
#     result_file.write(answer + '\n')


# write style in caption
# caption_file = '/mnt/petrelfs/liuwenran/datasets/changshiban/changshiban_ancient_frames_caption.txt'
# caption_file = open(caption_file, 'r')

# caption_lines = caption_file.readlines()
# style_caption_lines = []
# for line in caption_lines:
#     style_caption_lines.append('changshiban,' + line)

# style_caption_file = '/mnt/petrelfs/liuwenran/datasets/changshiban/changshiban_ancient_frames_caption_style.txt'
# style_caption_file = open(style_caption_file, 'w')
# style_caption_file.writelines(style_caption_lines)


# write jsonl file
import jsonlines

image_lines = open('/mnt/petrelfs/liuwenran/datasets/changshiban/changshiban_ancient_frames.txt', 'r').readlines()
caption_lines = open('/mnt/petrelfs/liuwenran/datasets/changshiban/changshiban_ancient_frames_caption_style.txt', 'r').readlines()

# 定义要写入的JSON数据列表
data = []
for ind, line in enumerate(image_lines):
    data.append({"file_name": line.strip().split('/')[-1], "text": caption_lines[ind].strip()})


# 打开文件，如果文件不存在则创建它
with jsonlines.open("/mnt/petrelfs/liuwenran/datasets/changshiban/train_dataset/metadata.jsonl", mode="w") as writer:
    # 循环写入每个JSON对象
    for item in data:
        writer.write(item)
