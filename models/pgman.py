"""PGMAN model definition."""

import re

import torch
import torch.nn as nn
from einops import rearrange
from transformers import BertTokenizer

from models.CModalT import classif_head
from models.at2 import AudioTransformer
from models.blip2qformer import MMCrossAttention
from models.decalign import DecAlign
from models.mbt_fusion import MBT
from models.text_encoder import TextEncoder
from models.vit import Vit

def pre_caption(caption: str, max_words: int = 20) -> str:
    """Normalize a generated caption and cap its length."""
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip("\n").strip()
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

class PGMAN(nn.Module):
    def __init__(
        self,
        num_frames: int = 8,
        sample_size: int = 224,
        n_classes: int = 8,
        need_audio: bool = True,
        need_text: bool = True,
        audio_embed_size: int = 256,
        audio_n_segments: int = 8,
        text_embed_size: int = 768,
    ) -> None:
        super().__init__()

        self.need_audio = need_audio
        self.need_text = need_text
        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size
        self.num_frames = num_frames
        self.n_classes = n_classes
        self.tsformer = Vit(
            img_size=sample_size,
            num_classes=600,
            num_frames=num_frames,
            attention_type='divided_space_time',
        )
        self.visual_embed_size = self.tsformer.model.embed_dim
        if need_text:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.textencoder = TextEncoder()

        self.auformer = AudioTransformer(
            audio_n_segments,
            segment_len=audio_embed_size,
            num_classes=n_classes,
            embed_dim=768,
            depth=2,
        )
        self.h = nn.Linear(self.visual_embed_size, self.visual_embed_size)
        self.model = MBT(2, 8, n_classes, 768)
        self.align_net = DecAlign(
            visual_dim=self.visual_embed_size,
            text_dim=text_embed_size,
            audio_dim=self.visual_embed_size,
            projection_dim=768,
        )
        self.cross = MMCrossAttention(layer_num=1)
        self.head2 = classif_head(self.visual_embed_size, n_classes, drop=0.5)

    def forward(
        self,
        visual: torch.Tensor,
        audio: list,
        text: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.need_audio or not self.need_text:
            raise ValueError("PGMAN requires both audio and cached captions")
        if not (visual.shape[0] == len(audio) == len(text)):
            raise ValueError("visual, audio, and text batch sizes must match")

        batch_size = visual.shape[0]
        with torch.no_grad():
            frame_batch = rearrange(
                visual,
                'b c t h w -> (b t) c h w',
                b=batch_size,
                t=self.num_frames,
            ).contiguous()
            visual = visual.contiguous()
            visual_cls, visual_tokens, frame_features = self.tsformer(
                visual, frame_batch
            )

        frame_features = self.h(frame_features)
        visual_frames = rearrange(
            frame_features,
            '(b t) c -> b t c',
            b=batch_size,
            t=self.num_frames,
        )

        audio_features = []
        text_features = []
        fused_features = []
        for index, (caption, audio_path) in enumerate(zip(text, audio)):
            with torch.no_grad():
                audio_feature = self.auformer(audio_path)
                tokens = self.text2tensor(caption, visual.device)
                text_feature = self.textencoder(
                    tokens["input_ids"], tokens["attention_mask"]
                )

            fused = self.cross(
                visual_frames[index].unsqueeze(0),
                t_encoder_hidden_states=text_feature,
                a_encoder_hidden_states=audio_feature.view(1, 1, -1),
            )
            audio_features.append(audio_feature)
            text_features.append(text_feature[0, 0])
            fused_features.append(fused[0, 0])

        audio_features = torch.stack(audio_features)
        text_features = torch.stack(text_features)
        fused_features = torch.stack(fused_features)

        alignment_loss = self.align_net(
            visual_cls, text_features, audio_features
        )
        bottleneck_logits = self.model(
            visual_tokens, audio_features.unsqueeze(1)
        )
        aligned_logits = self.head2(fused_features)
        logits = 0.2 * bottleneck_logits + 0.8 * aligned_logits
        return logits, alignment_loss

    def text2tensor(self, text: str, device: torch.device) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            pre_caption(text),
            return_tensors="pt",
            truncation=True,
            max_length=22,
        )
        return {name: value.to(device) for name, value in encoded.items()}
