import torch
import torch.nn as nn
import timm


class SwinTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        model_name='swin_base_patch4_window7_224',
        max_width,
        max_height,
        channels=1,
        pretrained=True,
        dropout=0.,
    ):
        super().__init__()

        # Initialize Swin Transformer from timm
        self.swin_transformer = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Ensure input channels match
        self.input_proj = nn.Conv2d(channels, self.swin_transformer.feature_info[0]['num_chs'], kernel_size=1)

        # Output projection to match encoder output size
        self.output_proj = nn.Linear(
            self.swin_transformer.feature_info[-1]['num_chs'], 
            self.swin_transformer.feature_info[-1]['num_chs']
        )
        self.dropout = nn.Dropout(dropout)
        self.max_width = max_width
        self.max_height = max_height

    def forward(self, img):
        # Adjust channels if input is grayscale
        img = self.input_proj(img)

        # Extract hierarchical features
        features = self.swin_transformer(img)

        # Use the last layer features
        x = features[-1]
        x = x.flatten(2).transpose(1, 2)  # Flatten spatial dimensions into sequence
        x = self.output_proj(x)
        return self.dropout(x)


def get_encoder(args):
    return SwinTransformerWrapper(
        model_name=args.get('swin_model_name', 'swin_base_patch4_window7_224'),
        max_width=args.max_width,
        max_height=args.max_height,
        channels=args.channels,
        pretrained=args.get('pretrained', True),
        dropout=args.get('dropout', 0.),
    )
