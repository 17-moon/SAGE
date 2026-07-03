import os
import numpy as np
import pickle
import _pickle as cPickle
import json
from collections import defaultdict
from os.path import join
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):

    def __init__(self, args, split):
        super().__init__()
        self.dataset = args.dataset
        self.split = split
        if split=='train':
            feat_path = 'train_{}_{}'.format(args.train_traj, args.clip_len)  
        elif split=='val':
            self.gt_rels = json.load(open(
                join('/home/xwy/MMP/', 'dataset', self.dataset, 'data', 'val_relation_gt.json'),"r"))
            feat_path = 'val_{}_{}'.format(args.val_traj, args.clip_len)
        elif split=='test':
            self.gt_rels = json.load(open('/data2/xwy/Data/vidvrd/data/test_relation_gt.json', 'r'))
            feat_path = 'test_{}_{}'.format(args.test_traj, args.clip_len)

        self.FEAT_ROOT = join('/data2/xwy/vidvrd/feature', feat_path)
        self.FEAT_ROOT = join(self.FEAT_ROOT, feat_path)
        # self.path_list = [[join(self.FEAT_ROOT, path), join(self.FEAT_ROOT+"_ptm", path), join(self.FEAT_ROOT+"_box", path)] for path in os.listdir(self.FEAT_ROOT)]
        
        print(f"Loading features from: {self.FEAT_ROOT}")

        
        self.path_list = [
            join(self.FEAT_ROOT, path) 
            for path in os.listdir(self.FEAT_ROOT) 
            if os.path.isfile(join(self.FEAT_ROOT, path))  # 只有是文件才加入列表
        ]
       
        DATA_ROOT = join('/data2/xwy/Data', self.dataset, 'data')
        self.id2pre = json.load(open(join(DATA_ROOT, 'id2predicate.json'), "r"))
        self.pre2id = json.load(open(join(DATA_ROOT, 'predicate2id.json'), "r"))
        self.id2obj = json.load(open(join(DATA_ROOT, 'id2object.json'), "r"))
        self.obj2id = json.load(open(join(DATA_ROOT, 'object2id.json'), "r"))
        
        self.pre_num = len(self.id2pre)
        
        # 3. 修改 prior.pkl 的加载路径
        self.prior = pickle.load(open(join(DATA_ROOT, 'prior.pkl'), 'rb'))
        if isinstance(self.id2pre, dict):
            # 如果是字典 {"0": "above", ...}
            self.pre_name_to_id = {v: int(k) for k, v in self.id2pre.items()}
        else:
            # 如果是列表 ["above", "away", ...]
            self.pre_name_to_id = {v: i for i, v in enumerate(self.id2pre)}

        # self.all_pair_data_rel = cPickle.load(open(join(self.FEAT_ROOT+'.pkl'),"rb"))
        # self.all_pair_data_ptm = cPickle.load(open(join(self.FEAT_ROOT+"_ptm.pkl"),"rb"))
        # self.all_pair_data_box = cPickle.load(open(join(self.FEAT_ROOT+"_box.pkl"),"rb"))

  

    def __getitem__(self, index):
        pair_path = self.path_list[index]
        filename = os.path.basename(pair_path).replace('.pkl', '')
        
        # 1. 修正 VID
        parts = filename.split('_')
        vid = "_".join(parts[:3]) 
        
        with open(pair_path, "rb") as f:
            data = pickle.load(f)
        
        # 2. 修正视觉特征维度 (核心：确保它是 512 维)
        raw_feat = np.array(data[0]['v2d_feat'], dtype=np.float32)
        seq_len = raw_feat.shape[0]
        
        if raw_feat.shape[-1] == 3072:
            # 原始是 4 份 768，我们将每一份都裁切到 512
            reshaped = raw_feat.reshape(seq_len, 4, 768)
            short_feat = reshaped[:, :, :512] # 每一份取前 512
            feat_final = short_feat.reshape(seq_len, -1) # 变成 (seq_len, 2048)
        else:
            # 如果已经是 2048 或者是别的，确保它最后是 2048
            feat_final = raw_feat[:, :2048]

        item = {'clip_feat': feat_final}

        item['rel_feat'] = np.zeros((seq_len, 64), dtype=np.float32)
        item['mot_feat'] = np.zeros((seq_len, 20), dtype=np.float32)
        
        # 3. 边界框特征：根据大部分 Baseline 的习惯，这里设为 4*24=96 维
        # 如果还报 96 的错，咱就看一眼 boxEmb 的输入维度
        item['bbox_feat'] = np.zeros((seq_len, 4, 24), dtype=np.float32)

        # 3. 匹配标签 (保持你现有的 triplet 逻辑)
        meta_info = data[1]
        start_f, end_f = meta_info['duration']
        pair_label = np.zeros((seq_len, self.pre_num))
        has_label = False
        
        if vid in self.gt_rels:
            relations = self.gt_rels[vid]
            for rel in relations:
                if isinstance(rel, dict) and 'triplet' in rel:
                    rel_s, rel_e = rel.get('duration', [0, 0])
                    pre_name = rel['triplet'][1]
                    if min(end_f, rel_e) > max(start_f, rel_s):
                        if pre_name in self.pre2id:
                            p_id = self.pre2id[pre_name]
                            pair_label[:, p_id] = 1
                            has_label = True

        # --- 4. 构建返回字典：严格遵守模型要求的维度 ---
        item = {}
        # 视觉主特征：必须是 512 维
        item['clip_feat'] = feat_final 

        # 空间相对特征：必须加起来等于 84 (64 + 20)
        item['rel_feat'] = np.zeros((seq_len, 64), dtype=np.float32)
        item['mot_feat'] = np.zeros((seq_len, 20), dtype=np.float32)

        # 边界框特征：通常模型 boxEmb 预期是 24 维
        item['bbox_feat'] = np.zeros((seq_len, 4, 24), dtype=np.float32)
        
        # 其他元数据
        item['pre_label'] = pair_label if has_label else None
        item['vid'] = vid
        item['action_feat'] = np.zeros((seq_len, 4, 512), dtype=np.float32)

        if has_label:
            print(f"✅ Ready! VID={vid}, Feat={item['clip_feat'].shape}, Label={pre_name}")
        return item

    def __len__(self):
        return len(self.path_list)

def padding_collate_fn(batch):
    seq_lens = torch.LongTensor([len(x['clip_feat']) for x in batch])
    batch_data = {}
    
    for k in batch[0]:
        if k in ['mask_feat', 'image_feat', 'vid', 'pair_data']: 
            continue
        
        # 收集 batch 中所有的值
        vals = [x[k] for x in batch]
        
        # --- 重点：处理 NoneType 防止报错 ---
        if k == 'pre_label':
            # 只有不为 None 的才转换成 Tensor，为空的给全 0 占位
            batch_data[k] = []
            for x in vals:
                if x is not None:
                    batch_data[k].append(torch.from_numpy(x).type(torch.float32))
                else:
                    # 如果这一个 sample 没标签，给一个全 0 的 Tensor
                    # 这里假设 seq_len 为 30
                    batch_data[k].append(torch.zeros((len(batch[0]['clip_feat']), 132)))
            batch_data[k] = pad_sequence(batch_data[k], batch_first=True)
            
        elif k in ['sbj_label', 'obj_label']:
            batch_data[k] = torch.tensor(vals).type(torch.long)
        else:
            # 处理 clip_feat, bbox_feat 等常规特征
            batch_data[k] = pad_sequence([torch.from_numpy(x).type(torch.float32) for x in vals], batch_first=True)
            
    return batch_data, seq_lens