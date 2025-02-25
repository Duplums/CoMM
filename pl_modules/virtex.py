import torch
from pl_modules.base import BaseModel


class VirTex(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = torch.hub.load("kdexd/virtex", "resnet50", pretrained=True)

    def extract_features(self, loader: torch.utils.data.DataLoader):
        """
           Extract global average pooled visual features.
           Args:
               loader: Dataset loader to serve ``(image, label)`` tuples.
            Returns: Pair (X,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for images, target in loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            with torch.inference_mode():
                # compute output
                output = self.forward(images)
                X.extend(output.view(len(output), -1).detach())
                y.extend(target.detach())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0), torch.stack(y, dim=0)

    def forward(self, x):
        b, c, h, w = x.shape
        z = self.encoder(x).view(b, 2048, h//32, w//32)
        # Apply 2D avg pooling to get 2048-d feature vector.
        z = torch.nn.functional.avg_pool2d(z, (h//32, w//32)).reshape(b, 2048)
        return z

    def test_step(self, batch, batch_idx):
        return None