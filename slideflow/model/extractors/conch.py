"""CONCH v1.5 feature extractor.

This module implements the CONCH v1.5 (CONtrastive learning from Captions for Histopathology)
feature extractor, which is distributed via the TITAN model on HuggingFace.

CONCH v1.5 is the patch encoder component of TITAN, designed for 512x512 pixel patches
at 20x magnification, but can also be used with 448x448 images.
"""

import torch
import torchvision.transforms as T
from typing import Optional

from ._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class ConchV15Features(TorchFeatureExtractor):
    """CONCH v1.5 pretrained feature extractor.

    CONCH v1.5 is the patch encoder component of TITAN (vision-language model for
    computational pathology). This implementation loads the model via the TITAN
    HuggingFace repository using the `return_conch()` method.

    The model uses a Vision Transformer backbone with attentional pooling to generate
    patch embeddings. Default input size is 448x448 pixels.

    Feature dimensions: 1024 (raw ViT CLS token, before attentional pooling)

    Manuscript: Lu, M. Y., et al. (2024). A visual-language foundation model for
    computational pathology. Nature Medicine.

    HuggingFace: https://huggingface.co/MahmoodLab/TITAN

    """

    tag = 'conch_v1.5'
    license = """CC-BY-NC-ND 4.0 (non-commercial use only). Please see the original license at https://huggingface.co/MahmoodLab/TITAN."""
    citation = """
@article{lu2024visual,
  title={A visual-language foundation model for computational pathology},
  author={Lu, Ming Y and Chen, Bowen and Williamson, Drew FK and Chen, Richard J and Zhao, Melissa and Chow, Aaron K and Ikemura, Kenji and Kim, Ahrong and Pouli, Dimitra and Patel, Ankush and others},
  journal={Nature Medicine},
  pages={1--11},
  year={2024},
  publisher={Nature Publishing Group}
}
"""

    def __init__(self, device=None, **kwargs):
        """Initialize CONCH v1.5 feature extractor.

        Args:
            device: Device to use for inference ('cuda', 'cpu', or None for auto-detect)
            **kwargs: Additional arguments passed to TorchFeatureExtractor
        """
        super().__init__(**kwargs)

        from slideflow.model import torch_utils
        from transformers import AutoModel

        self.device = torch_utils.get_device(device)

        # Load TITAN and extract CONCH v1.5 using the official method
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        self._visual_model, self.eval_transform = titan.return_conch()

        self._visual_model.to(self.device)
        self._visual_model.eval()

        # ---------------------------------------------------------------------
        # CONCH v1.5 Vision Transformer has 1024-dim embeddings (context_dim)
        # The 768-dim mentioned in docs is after attentional pooling for contrastive learning
        # For downstream tasks, we use the raw 1024-dim CLS token
        self.num_features = 1024

        # CONCH v1.5 uses 448x448 images with specific normalization
        # The eval_transform from TITAN expects PIL/numpy and includes ToTensor()
        # Since Slideflow provides torch tensors (uint8, WHC), we need to build our own transform
        # that matches CONCH's preprocessing without the ToTensor step
        self.img_size = 448

        # CONCH uses ImageNet normalization (OPENAI_DATASET_MEAN/STD from open_clip)
        # mean = [0.48145466, 0.4578275, 0.40821073]
        # std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            T.Lambda(lambda x: x / 255.),  # Convert uint8 to [0, 1] float
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        self.preprocess_kwargs = dict(standardize=False)

    def model(self, x):
        """Extract features before attentional pooling and normalization.

        For image-only downstream tasks, features should be extracted from the
        vision transformer backbone before the attentional pooling and contrastive
        projection. This extracts the CLS token from the final transformer layer.

        This matches the official CONCH usage pattern:
        encode_image(img, proj_contrast=False, normalize=False)
        """
        # Extract all tokens from the vision transformer
        x = self._visual_model.trunk(x, return_all_tokens=True)
        # Return the CLS token (first token) - shape: (batch, 1024)
        return x[:, 0]

    def _process_output(self, output):
        """Process model output to extract embeddings."""
        return output.to(torch.float32)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.conch.ConchV15Features'
        )


# Alias for convenience
class ConchFeatures(ConchV15Features):
    """Alias for ConchV15Features (defaults to CONCH v1.5)."""
    tag = 'conch'
