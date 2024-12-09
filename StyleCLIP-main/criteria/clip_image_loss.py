import torch
import clip


class CLIPImageLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPImageLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image1, image2):
        image1 = self.avg_pool(self.upsample(image1))
        image2 = self.avg_pool(self.upsample(image2))

        # Compute embeddings
        image1_features = self.model.encode_image(image1)
        image2_features = self.model.encode_image(image2)

        # Normalize embeddings
        image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
        image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity
        similarity = torch.cosine_similarity(image1_features, image2_features)
        return 1 - similarity.mean()