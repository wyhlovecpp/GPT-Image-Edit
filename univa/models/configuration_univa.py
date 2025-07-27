from transformers import Qwen2Config
from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from typing import Optional


class UnivaConfig(Qwen2Config):
    model_type = "univa"
    sub_configs = {
        "denoise_tower": UnivaDenoiseTowerConfig,
    }

    def __init__(
        self,
        denoise_tower: UnivaDenoiseTowerConfig = None,
        image_token_length: Optional[int] = None,
        shortcut_image_embeds: bool = False,
        shortcut_image_embeds_scale: float = 0.5,
        shortcut_projector_type: Optional[str] = "mlp2x_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_length = image_token_length
        self.shortcut_image_embeds = shortcut_image_embeds
        self.shortcut_image_embeds_scale = shortcut_image_embeds_scale

        if not shortcut_image_embeds:
            shortcut_projector_type = None


        if isinstance(denoise_tower, dict):
            denoise_tower["input_hidden_size"] = self.hidden_size
            self.denoise_tower = UnivaDenoiseTowerConfig(**denoise_tower)
        elif denoise_tower is None:
            self.denoise_tower = UnivaDenoiseTowerConfig(
                input_hidden_size=self.hidden_size
            )
        else:
            self.denoise_tower = denoise_tower
