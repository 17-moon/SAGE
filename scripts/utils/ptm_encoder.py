from clip import clip
import InternVideo
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from InternVideo import video_transform
# import cv2
# from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration


class CLIPVisualEncoder(nn.Module):
    def __init__(self, backbone="ViT-B/16", gpu_id=0):
        super().__init__()
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load(name = backbone, device = self.device)
        self.preprocess = preprocess
        self.visual =  clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, regions):
        feats = []
        for image_path, bbox in regions:
            if bbox:
                image = Image.open(image_path).crop(bbox)
            else:
                image = Image.open(image_path)
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            feat = self.visual(image.type(self.dtype)).detach().cpu().numpy()[0]
            feats.append(feat)
        return np.asarray(feats)

class ACLIPVisualEncoder(nn.Module):
    def __init__(self, backbone="ViT-B/16", gpu_id=0):
        super().__init__()
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load(name = backbone, device = self.device)
        self.visual =  clip_model.visual
        self.dtype = clip_model.dtype
        self.model, self.preprocess = alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth="/home/xwy/MMP_OV_VidVRD-main/clip_b16_grit+mim_fultune_4xe.pth", device=self.device )  # change to your own ckpt path
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((224, 224),antialias=True), # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize(0.5, 0.26)
        ])
        

    def forward(self, regions):
        feats = []
        a = 1
        for image_path, bbox in regions:
            if bbox :
                image = Image.open(image_path).crop(bbox)
                mask_pth=str(image_path)[:-4]+'_mask'+str(a)+'.png'
                mask = Image.open(mask_pth)
                mask = np.array(mask) 

          
                if len(mask.shape) == 2: binary_mask = (mask == 255)
                if len(mask.shape) == 3: binary_mask = (mask[:, :, 0] == 255)

                alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
                alpha = alpha.half().cuda().unsqueeze(dim=0)

                # calculate image and text features
                image = self.preprocess(image).unsqueeze(0).half().to(self.device)

                with torch.no_grad():
                    image_features = self.model.visual(image, alpha).detach().cpu().numpy()[0]
                    feats.append(image_features)
                    a+=1
            else :
                
                image = Image.open(image_path)
                image = self.preprocess(image).unsqueeze(0).to(self.device)
                feat = self.visual(image.type(self.dtype)).detach().cpu().numpy()[0]
                feats.append(feat)
        return np.asarray(feats)

class CLIPTransformer(nn.Module):
    def __init__(self, backbone="ViT-B/16", gpu_id=0):
        super().__init__()
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load(name = backbone, device = self.device)
        self.preprocess = preprocess
        
        self.dtype = clip_model.dtype

    def forward(self, regions):
        feats=[]
        for image_path, bbox in regions:
            if bbox:
                image = Image.open(image_path).crop(bbox)
            else:
                image = Image.open(image_path)
            image = self.preprocess(image).to(self.device)

            feats.append(image.cpu().numpy())
            # feats.append({'image_path':image_path,'bbox':bbox})
           
        return np.asarray(feats)

class InternlVisualEncoder(nn.Module):
    def __init__(self, backbone="ViT-B/16", gpu_id=0):
        super().__init__()
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model = InternVideo.load_model("/home/xwy/MMP_OV_VidVRD-main/InternVideo-MM-B-16.ckpt").cuda(device=self.device)
        self.preprocess = transforms.Compose([
        video_transform.TensorToNumpy(),
        video_transform.Resize((224, 224)),
        video_transform.ClipToTensor(channel_nb=3),
        video_transform.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    def forward(self, sequences):
        feats = []
        for sequence in sequences:
            video = []
            for image_path, bbox in zip(sequence[0], sequence[1]):
                if bbox:
                    image = Image.open(image_path).crop(bbox).resize((224,224))
                else:
                    image = Image.open(image_path)
                video.append(torch.from_numpy(np.array(image)).unsqueeze(0))
            video = torch.cat(video, dim=0)
            video = video[np.linspace(0, len(video) - 1, 8).astype(np.int64)]
            video = video.permute(3, 0, 1, 2)
            video = self.preprocess(video).unsqueeze(0).to(self.device)
            x=self.model.encode_video(video).detach().cpu().numpy()[0]
            feats.append(self.model.encode_video(video).detach().cpu().numpy()[0])

        return np.asarray(feats)
    

class BilpTransformer(nn.Module):
    def __init__(self, gpu_id=0):
        super().__init__()
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.model.eval()
        self.model.to(self.device)
        
        self.clip_model, preprocess = clip.load(name = 'ViT-L/14', device = self.device)
        self.clip_model=self.clip_model.eval()
       
    def forward(self, regions,prompt):
        feats=[]

        for image_path, bbox in regions:
            if bbox:
                image = Image.open(image_path).crop(bbox)
            else:
                image = Image.open(image_path)
          
            inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs)
            # generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
           
            prompts = clip.tokenize(generated_text).cuda()
            with torch.no_grad():
                text_embeddings = self.clip_model.encode_text(prompts)
                # text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            feats.append(text_embeddings.cpu().numpy())
        return np.asarray(feats)

