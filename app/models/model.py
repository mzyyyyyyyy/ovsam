import torch.nn.functional as F
from mmengine.model import BaseModel

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class SAMSegmentor(BaseModel):
    MASK_THRESHOLD = 0.5

    def __init__(
            self,
            backbone: ConfigType,
            neck: ConfigType,
            prompt_encoder: ConfigType,
            mask_decoder: ConfigType,
            data_preprocessor: OptConfigType = None,
            fpn_neck: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            use_clip_feat: bool = False,
            use_head_feat: bool = False,
            use_gt_prompt: bool = False,
            use_point: bool = False,
            enable_backbone: bool = False,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.pe = MODELS.build(prompt_encoder)
        self.mask_decoder = MODELS.build(mask_decoder)
        if fpn_neck is not None:
            self.fpn_neck = MODELS.build(fpn_neck)
        else:
            self.fpn_neck = None

        self.use_clip_feat = use_clip_feat
        self.use_head_feat = use_head_feat
        self.use_gt_prompt = use_gt_prompt
        self.use_point = use_point

        self.enable_backbone = enable_backbone

    def extract_feat(self, inputs):
        backbone_feat = self.backbone(inputs) # 四个元素，shape 分别为(1, 384, 256, 256)(1, 768, 128, 128)(1, 1536, 64, 64)(1, 3072, 32, 32)
        neck_feat = self.neck(backbone_feat) # (1, 256, 64, 64), 这是一个多层 Transformer，它就是本方法训练好的学生模型。
        if self.fpn_neck is not None:
            fpn_feat = self.fpn_neck(backbone_feat) # 四个元素，shape 分别为(1, 256, 256, 256)(1, 256, 128, 128)(1, 256, 64, 64)(1, 256, 32, 32)，这个模块也是可以训练的。
        else:
            fpn_feat = None

        return dict(
            backbone_feat=backbone_feat,
            neck_feat=neck_feat,
            fpn_feat=fpn_feat
        )

    def extract_masks(self, feat_cache, prompts):
        sparse_embed, dense_embed = self.pe(
            prompts,
            image_size=(1024, 1024),
            with_points='point_coords' in prompts,
            with_bboxes='bboxes' in prompts,
        )# torch.Size([1, 2, 256]) torch.Size([1, 256, 64, 64]) 这就是 prompt encoder.

        kwargs = dict()
        if self.enable_backbone:
            kwargs['backbone_feats'] = feat_cache['backbone_feat']
            kwargs['backbone'] = self.backbone
            kwargs['fpn_feats'] = feat_cache['fpn_feat']
        low_res_masks, iou_predictions, cls_pred = self.mask_decoder(
            image_embeddings=feat_cache['neck_feat'],
            image_pe=self.pe.get_dense_pe(), 
            # 这个也许是图像的位置编码？答：是的，不用细究细节，就是已经预存储好的不可学习的参数。
            sparse_prompt_embeddings=sparse_embed,
            dense_prompt_embeddings=dense_embed,
            multi_mask_output=False,
            **kwargs
        ) # torch.Size([1, 1, 256, 256]) torch.Size([1, 1, 1204])
        masks = F.interpolate(
            low_res_masks,
            scale_factor=4.,
            mode='bilinear',
            align_corners=False,
        ) # torch.Size([1, 1, 1024, 1024])

        masks = masks.sigmoid()
        cls_pred = cls_pred.softmax(-1)[..., :-1] # torch.Size([1, 1, 1203])
        return masks.detach().cpu().numpy(), cls_pred.detach().cpu()

    def forward(self, inputs):
        return inputs
