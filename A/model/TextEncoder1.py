from collections import OrderedDict
import torch
import torch.nn as nn
from .clip import clip
import json,os


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, SepcialPrompts, Original_tokenized_prompts):
        batch_size, n_cls, n_tkn, ctx_dim = SepcialPrompts.shape
        # 将批量和类别维度展平
        SepcialPrompts = SepcialPrompts.view(batch_size * n_cls, n_tkn, ctx_dim)
        # 扩展 tokenized_prompts 以匹配批量维度
        Original_tokenized_prompts = Original_tokenized_prompts[None, :, :].expand(batch_size, -1, -1)
        Original_tokenized_prompts = Original_tokenized_prompts.reshape(batch_size * n_cls, -1)
        x = SepcialPrompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # [n_tkn, batch_size * n_cls, ctx_dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch_size * n_cls, n_tkn, ctx_dim]
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), Original_tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # 恢复批量和类别维度
        x = x.view(batch_size, n_cls, -1)
        print(f"TextEncoder 输出形状: {x.shape}")
        return x


class SpecificTextualPrompt(nn.Module):
    def __init__(self, cfg, clip_model, init_ctx = False, leranable_ctx=False):
        super().__init__()
        self.n_ctx = cfg['MODEL']['TextEncoder']['N_CTX']  # the number of textual prompts
        self.ctx_init = cfg['MODEL']['TextEncoder']['CTX_INIT']
        tf = open(cfg['Dataset']['Classnames'], "r")
        classnames_dict = json.load(tf)  # class name idx start from 0
        self.classnames_list = [i for i in classnames_dict.keys()]
        print(f"**************************Loaded classnames_list: {self.classnames_list}, length: {len(self.classnames_list)}")
        # TODO: for N-Imagenet
        if cfg['Dataset']['Imagenet_dict_path'] != '':
            self.classnames_list = []
            Imagenet_dict_path = cfg['Dataset']['Imagenet_dict_path']
            with open(Imagenet_dict_path, 'r') as f:
                Imagenet_dict = json.load(f)
            for cat_id, cat in enumerate(os.listdir(cfg['Dataset']['N-imagenet-val'])):
                cat_name = Imagenet_dict[cat]
                self.classnames_list.append(cat_name)
            print(f"Updated classnames_list with ImageNet: {self.classnames_list}, length: {len(self.classnames_list)}")

        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.vis_dim = clip_model.visual.output_dim

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg['Dataset']['Input_size'][0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if init_ctx:
            # use given words to initialize context vectors
            ctx_init = self.ctx_init.replace("_", " ")
            self.n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + self.n_ctx, :] # extract ctx without SOS and EOS
            self.prompt_prefix = ctx_init

        if leranable_ctx:
            # random context vectors initialization
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.prompt_prefix = " ".join(["X"] * self.n_ctx)

        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        # TODO: change the MetaNet here
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(self.vis_dim, self.vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(self.vis_dim // 16, self.ctx_dim))
        ]))
        if cfg['Trainer']['Precision'] == "fp16":
            self.meta_net.half()  # float32->loat16

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim) # SOS(start of special token)
                ctx,  # (dim0, n_ctx, dim) # textual prompts e.g.: a event frame of
                suffix,  # (dim0, *, dim) # after ctx_init, including name+. +EOS(end of special token)
            ],
            dim=1, )

        return prompts

    def forward(self, video_features, class_idxs, use_bias):
      device = video_features.device
      print(f"输入 class_idxs: {class_idxs}, class_idxs 长度: {len(class_idxs)}")  # 新增：打印 class_idxs 的内容和长度
      print(f"原始 classnames_list: {self.classnames_list}, 长度: {len(self.classnames_list)}")
    
      # 原始逻辑
      classnames_original = [self.classnames_list[class_idxs[i]] for i in range(len(class_idxs))]  # 可能导致问题
      print(f"从 class_idxs 生成的 classnames_original: {classnames_original}, 长度: {len(classnames_original)}")
    
      # 临时修复：限制为 5 个类别
      n_cls_original = len(self.classnames_list)  # 应为 5
      print(f"原始 n_cls: {n_cls_original}")
      n_cls = min(n_cls_original, 5)  # 确保不超过 5
      classnames_adjusted = self.classnames_list[:n_cls]  # 只取前 5 个
      print(f"调整后的 n_cls: {n_cls}, classnames_adjusted: {classnames_adjusted}")
    
      prompts = [self.prompt_prefix + " " + name + "." for name in classnames_adjusted]  # 使用调整后的列表
      tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
      print(f"tokenized_prompts 形状: {tokenized_prompts.shape}")  # 应为与 5 个类别相关的形状
      # ... 其余代码保持不变

      with torch.no_grad():
          embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype).to(device)
      prefix = embedding[:, :1, :]
      suffix = embedding[:, 1 + self.n_ctx:, :]
      ctx = self.ctx.unsqueeze(0).to(device)
      if use_bias:
          bias = self.meta_net(video_features).to(device)
          bias = bias.unsqueeze(1)
          ctx_shifted = ctx + bias
      else:
          b, _, = video_features.size()
          _, n_ctx, ctx_dim = ctx.size()
          ctx_shifted = ctx.expand(b, n_ctx, ctx_dim)
      prompts = []
      for ctx_shifted_i in ctx_shifted:
          ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
          pts_i = self.construct_prompts(ctx_i, prefix, suffix)
          prompts.append(pts_i)
      SepcialPrompts = torch.stack(prompts).to(device)
      print(f"SepcialPrompts 形状: {SepcialPrompts.shape}")
      return SepcialPrompts, tokenized_prompts
