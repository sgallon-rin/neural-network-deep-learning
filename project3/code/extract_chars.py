""" 提取出训练集所有出现过的字符 """
import os
from tqdm import tqdm
from config import HOME, DATA_ROOT

S = set()

# train_img_path = os.path.join(DATA_ROOT, "train", "img")
train_gt_path = os.path.join(DATA_ROOT, "train", "gt")

for gt in tqdm(os.listdir(train_gt_path)):
    if not gt.endswith(".txt"):
        continue
    with open(os.path.join(train_gt_path, gt), "r") as f:
        for line in f:  # 一行是“x1,y1,x2,y2,x3,y3,x4,y4,语言,内容”
            text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
            transcript = text[9]  # 只要内容，不管语言
            for s in transcript:  # 将出现过的字符逐一加入集合
                S.add(s)

S = ''.join(sorted(list(S)))

with open(os.path.join(HOME, "code", "vocab.txt"), 'w', encoding='utf-8') as f:
    f.write(S)
