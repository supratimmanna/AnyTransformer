import torch
import torch.nn as nn

class Patch_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config['image_size'] % config['patch_size'] == 0, 'Image size must be divisible by patch size'
      
        self.num_patches = int((config['image_size']/config['patch_size'])**2)

        self.patchify_layer = nn.Conv2d(in_channels=config['image_channel'], out_channels=config['embed_dim'], kernel_size=config['patch_size'],
                           stride=config['patch_size'])
        
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)

        self.class_token_embeddig = nn.Parameter(torch.rand(1, 1, config['embed_dim']))

        self.pos_embedding = nn.Parameter(torch.rand(1, self.num_patches+1, config['embed_dim']))
        self.dropout = nn.Dropout(config['input_embed_dropout'])
        

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patchify_layer(x).permute(0, 2, 3, 1)
        x = self.flatten_layer(x)
        
        ## adding class token and then add postitional embedding
        class_token_embeddig = self.class_token_embeddig.expand(batch_size, -1, -1)
        x_with_class = torch.cat((class_token_embeddig, x), dim=1)
        input_embedding = x_with_class + self.pos_embedding

        input_embedding = self.dropout(input_embedding)

        return input_embedding