from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from face_alignment.models import FAN
from torchvision.models import ResNet18_Weights, resnet18


def _round_channels(channels: int, width_mult: float) -> int:
    return int((channels * width_mult + 7) // 8 * 8)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.layers(inputs)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int, use_se: bool) -> None:
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        layers: list[nn.Module] = [ConvBNAct(in_channels, hidden_channels, 1)]
        layers.append(ConvBNAct(hidden_channels, hidden_channels, 3, stride=stride, groups=hidden_channels))
        if use_se:
            layers.append(SqueezeExcitation(hidden_channels))
        layers.extend(
            [
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.post_act = nn.ReLU(inplace=True)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.block(inputs)
        if self.use_residual:
            outputs = outputs + inputs
        return self.post_act(outputs)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvBNAct(in_channels, out_channels, 3),
            ConvBNAct(out_channels, out_channels, 3),
        )


class UNetHeatmapNet(nn.Module):
    def __init__(self, *, base_channels: int = 48, num_landmarks: int = 68) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_temperature = 8.0
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 6]
        self.stem = DoubleConv(3, channels[0])
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(channels[0], channels[1])
        self.encoder3 = DoubleConv(channels[1], channels[2])
        self.bottleneck = DoubleConv(channels[2], channels[3])

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(channels[0] * 2, channels[0])

        self.reduce = ConvBNAct(channels[0], channels[0], 3)
        self.heatmap_head = nn.Conv2d(channels[0], num_landmarks, 1)


    def _heatmaps_to_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, height, width = heatmaps.shape
        logits = heatmaps.view(batch_size, num_points, -1) * self.heatmap_temperature
        probabilities = torch.softmax(logits, dim=-1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=heatmaps.device),
            torch.linspace(0.0, 1.0, width, device=heatmaps.device),
            indexing="ij",
        )
        coord_x = (probabilities * grid_x.reshape(1, 1, -1)).sum(dim=-1)
        coord_y = (probabilities * grid_y.reshape(1, 1, -1)).sum(dim=-1)
        return torch.stack([coord_x, coord_y], dim=-1)


    def forward_train(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor]:
        skip1 = self.stem(inputs)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        bottleneck = self.bottleneck(self.pool(skip3))

        up3 = self.up3(bottleneck)
        dec3 = self.decoder3(torch.cat([up3, skip3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.decoder2(torch.cat([up2, skip2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.decoder1(torch.cat([up1, skip1], dim=1))

        heatmaps = self.heatmap_head(self.reduce(dec1))
        landmarks = self._heatmaps_to_coordinates(heatmaps)
        return landmarks, None, heatmaps


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        landmarks, _, _ = self.forward_train(inputs)
        return landmarks


class FANHeatmapNet(nn.Module):
    def __init__(self, *, num_modules: int = 2, num_landmarks: int = 68) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_temperature = 8.0
        self.fan = FAN(num_modules=num_modules)


    def _heatmaps_to_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size, num_points, height, width = heatmaps.shape
            logits = heatmaps.view(batch_size, num_points, -1) * self.heatmap_temperature
            probabilities = torch.softmax(logits, dim=-1)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0.0, 1.0, height, device=heatmaps.device),
                torch.linspace(0.0, 1.0, width, device=heatmaps.device),
                indexing="ij",
            )
            coord_x = (probabilities * grid_x.reshape(1, 1, -1)).sum(dim=-1)
            coord_y = (probabilities * grid_y.reshape(1, 1, -1)).sum(dim=-1)
            return torch.stack([coord_x, coord_y], dim=-1)

        # Inference: argmax + sub-pixel offset (matches face_alignment decoding)
        batch_size, num_points, height, width = heatmaps.shape
        flat = heatmaps.view(batch_size, num_points, -1)
        idx = flat.argmax(dim=-1)
        cell_y = torch.div(idx, width, rounding_mode="floor")
        cell_x = idx % width
        coord_x = cell_x.to(heatmaps.dtype)
        coord_y = cell_y.to(heatmaps.dtype)

        cx = cell_x.clamp(min=1, max=width - 2)
        cy = cell_y.clamp(min=1, max=height - 2)
        valid_x = (cell_x > 0) & (cell_x < width - 1)
        valid_y = (cell_y > 0) & (cell_y < height - 1)
        b_index = torch.arange(batch_size, device=heatmaps.device).view(-1, 1).expand(-1, num_points)
        p_index = torch.arange(num_points, device=heatmaps.device).view(1, -1).expand(batch_size, -1)
        right = heatmaps[b_index, p_index, cy, (cx + 1).clamp(max=width - 1)]
        left = heatmaps[b_index, p_index, cy, (cx - 1).clamp(min=0)]
        down = heatmaps[b_index, p_index, (cy + 1).clamp(max=height - 1), cx]
        up = heatmaps[b_index, p_index, (cy - 1).clamp(min=0), cx]
        coord_x = coord_x + 0.25 * torch.sign(right - left) * valid_x.to(heatmaps.dtype)
        coord_y = coord_y + 0.25 * torch.sign(down - up) * valid_y.to(heatmaps.dtype)
        coord_x = coord_x / max(width - 1, 1)
        coord_y = coord_y / max(height - 1, 1)
        return torch.stack([coord_x, coord_y], dim=-1)


    def forward_train(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor]:
        # Data pipeline outputs [-1,1]; FAN pretrained weights expect [0,1] inputs.
        fan_inputs = inputs * 0.5 + 0.5
        heatmap_stacks = self.fan(fan_inputs)
        heatmaps = heatmap_stacks[-1][:, : self.num_landmarks]
        landmarks = self._heatmaps_to_coordinates(heatmaps)
        return landmarks, None, heatmaps


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        landmarks, _, _ = self.forward_train(inputs)
        return landmarks


class ResNetDeconvHeatmapNet(nn.Module):
    def __init__(self, *, num_landmarks: int = 68, pretrained_backbone: bool = True, deconv_dim: int = 256) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_temperature = 8.0
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, deconv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(deconv_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(deconv_dim, deconv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(deconv_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(deconv_dim, deconv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(deconv_dim),
            nn.ReLU(inplace=True),
        )
        self.refine = ConvBNAct(deconv_dim, deconv_dim, 3)
        self.heatmap_head = nn.Conv2d(deconv_dim, num_landmarks, 1)


    def _heatmaps_to_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, height, width = heatmaps.shape
        logits = heatmaps.view(batch_size, num_points, -1) * self.heatmap_temperature
        probabilities = torch.softmax(logits, dim=-1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=heatmaps.device),
            torch.linspace(0.0, 1.0, width, device=heatmaps.device),
            indexing="ij",
        )
        coord_x = (probabilities * grid_x.reshape(1, 1, -1)).sum(dim=-1)
        coord_y = (probabilities * grid_y.reshape(1, 1, -1)).sum(dim=-1)
        return torch.stack([coord_x, coord_y], dim=-1)


    def forward_train(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor]:
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.deconv(outputs)
        heatmaps = self.heatmap_head(self.refine(outputs))
        landmarks = self._heatmaps_to_coordinates(heatmaps)
        return landmarks, None, heatmaps


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        landmarks, _, _ = self.forward_train(inputs)
        return landmarks


class LandmarkTranslationWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, *, num_landmarks: int, hidden_dim: int = 256, residual_scale: float = 0.15) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_landmarks = num_landmarks
        self.residual_scale = residual_scale
        self.adapter = nn.Sequential(
            nn.Linear(num_landmarks * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_landmarks * 2),
        )


    def forward_train(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        landmarks, pose, aux = self.base_model.forward_train(inputs)
        delta = self.adapter(landmarks.flatten(1)).view(-1, self.num_landmarks, 2)
        translated = (landmarks + self.residual_scale * torch.tanh(delta)).clamp(0.0, 1.0)
        return translated, pose, aux


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        landmarks, _, _ = self.forward_train(inputs)
        return landmarks


class LMNet(nn.Module):
    def __init__(
        self,
        *,
        width_mult: float = 1.75,
        hidden_dim: int = 768,
        dropout: float = 0.15,
        use_se: bool = True,
        num_landmarks: int = 68,
        head_type: str = "multiscale",
        fusion_dim: int = 160,
        pose_head: bool = False,
        residual_mean_shape: bool = False,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.head_type = head_type
        self.use_pose_head = pose_head
        self.heatmap_temperature = 8.0
        self.residual_mean_shape = residual_mean_shape
        self.residual_scale = residual_scale
        self.register_buffer("mean_shape", torch.full((num_landmarks, 2), 0.5), persistent=False)
        channels = [_round_channels(value, width_mult) for value in [24, 32, 64, 96, 160, 224]]
        self.stem = ConvBNAct(3, channels[0], 3, stride=2)
        self.stage1 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], 1, expand_ratio=2, use_se=False),
            InvertedResidual(channels[1], channels[1], 1, expand_ratio=2, use_se=False),
        )
        self.stage2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], 2, expand_ratio=3, use_se=False),
            InvertedResidual(channels[2], channels[2], 1, expand_ratio=3, use_se=False),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], 2, expand_ratio=4, use_se=use_se),
            InvertedResidual(channels[3], channels[3], 1, expand_ratio=4, use_se=use_se),
        )
        self.stage4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], 2, expand_ratio=4, use_se=use_se),
            InvertedResidual(channels[4], channels[4], 1, expand_ratio=4, use_se=use_se),
        )
        self.stage5 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], 2, expand_ratio=4, use_se=use_se),
            InvertedResidual(channels[5], channels[5], 1, expand_ratio=4, use_se=use_se),
            ConvBNAct(channels[5], channels[5], 1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if head_type == "pip":
            pip_dim = max(fusion_dim, 160)
            self.fusion_blocks = nn.ModuleList([])
            self.pip_refine = nn.Sequential(
                ConvBNAct(channels[2], pip_dim, 3),
                ConvBNAct(pip_dim, pip_dim, 3),
            )
            self.pip_cls_head = nn.Conv2d(pip_dim, num_landmarks, 1)
            self.pip_x_head = nn.Conv2d(pip_dim, num_landmarks, 1)
            self.pip_y_head = nn.Conv2d(pip_dim, num_landmarks, 1)
            landmark_input_dim = 0
            pose_input_dim = channels[5]
        elif head_type == "deconv_heatmap":
            deconv_dim = max(fusion_dim, 192)
            self.fusion_blocks = nn.ModuleList([])
            self.heatmap_refine = nn.Sequential(
                nn.ConvTranspose2d(channels[5], deconv_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(deconv_dim),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(deconv_dim, deconv_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(deconv_dim),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(deconv_dim, deconv_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(deconv_dim),
                nn.ReLU(inplace=True),
                ConvBNAct(deconv_dim, deconv_dim, 3),
            )
            self.heatmap_head = nn.Conv2d(deconv_dim, num_landmarks, 1)
            landmark_input_dim = 0
            pose_input_dim = channels[5]
        elif head_type == "heatmap_hr":
            stage_channels = max(fusion_dim // 2, 64)
            self.fusion_blocks = nn.ModuleList(
                [
                    ConvBNAct(channels[2], stage_channels, 1),
                    ConvBNAct(channels[3], stage_channels, 1),
                    ConvBNAct(channels[4], stage_channels, 1),
                    ConvBNAct(channels[5], stage_channels, 1),
                ]
            )
            self.heatmap_refine = nn.Sequential(
                ConvBNAct(stage_channels * 4, fusion_dim, 3),
                ConvBNAct(fusion_dim, fusion_dim, 3),
                ConvBNAct(fusion_dim, fusion_dim, 3),
            )
            self.heatmap_head = nn.Conv2d(fusion_dim, num_landmarks, 1)
            landmark_input_dim = 0
            pose_input_dim = channels[5]
        elif head_type == "heatmap":
            self.fusion_blocks = nn.ModuleList(
                [
                    ConvBNAct(channels[2], fusion_dim, 1),
                    ConvBNAct(channels[3], fusion_dim, 1),
                    ConvBNAct(channels[4], fusion_dim, 1),
                    ConvBNAct(channels[5], fusion_dim, 1),
                ]
            )
            self.heatmap_refine = nn.Sequential(
                ConvBNAct(fusion_dim, fusion_dim, 3),
                ConvBNAct(fusion_dim, fusion_dim, 3),
            )
            self.heatmap_head = nn.Conv2d(fusion_dim, num_landmarks, 1)
            landmark_input_dim = 0
            pose_input_dim = channels[5]
        elif head_type == "spatial_fusion":
            projection_channels = [fusion_dim // 5, fusion_dim // 5, fusion_dim * 3 // 10, fusion_dim * 3 // 10]
            self.grid_pool = nn.AdaptiveAvgPool2d((6, 6))
            self.fusion_blocks = nn.ModuleList(
                [
                    ConvBNAct(channels[2], projection_channels[0], 1),
                    ConvBNAct(channels[3], projection_channels[1], 1),
                    ConvBNAct(channels[4], projection_channels[2], 1),
                    ConvBNAct(channels[5], projection_channels[3], 1),
                ]
            )
            landmark_input_dim = sum(projection_channels) * 6 * 6
            pose_input_dim = channels[5]
        elif head_type == "multiscale":
            self.fusion_blocks = nn.ModuleList(
                [
                    ConvBNAct(channels[2], fusion_dim, 1),
                    ConvBNAct(channels[3], fusion_dim, 1),
                    ConvBNAct(channels[4], fusion_dim, 1),
                    ConvBNAct(channels[5], fusion_dim, 1),
                ]
            )
            landmark_input_dim = fusion_dim * 4
            pose_input_dim = fusion_dim * 2
        else:
            self.fusion_blocks = nn.ModuleList([nn.Identity()])
            landmark_input_dim = channels[5]
            pose_input_dim = channels[5]

        if head_type in {"heatmap", "heatmap_hr", "deconv_heatmap", "pip"}:
            self.landmark_head = None
        else:
            self.landmark_head = nn.Sequential(
                nn.Linear(landmark_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_landmarks * 2),
            )
        if pose_head:
            self.pose_head = nn.Sequential(
                nn.Linear(pose_input_dim, max(hidden_dim // 2, 128)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(max(hidden_dim // 2, 128), 3),
            )
        else:
            self.pose_head = None


    def initialize_landmark_bias(self, mean_shape: torch.Tensor) -> None:
        self.mean_shape.copy_(mean_shape)
        if self.landmark_head is None:
            return
        last_layer = self.landmark_head[-1]
        if not isinstance(last_layer, nn.Linear):
            return
        mean_shape = mean_shape.clamp(1e-4, 1 - 1e-4).reshape(-1)
        bias = torch.log(mean_shape / (1 - mean_shape))
        with torch.no_grad():
            last_layer.bias.copy_(bias)


    def extract_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.stem(inputs)
        outputs = self.stage1(outputs)
        stage2 = self.stage2(outputs)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        return stage2, stage3, stage4, stage5


    def _pooled_features(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head_type == "spatial_fusion":
            projected = [
                self.grid_pool(block(feature))
                for block, feature in zip(self.fusion_blocks, features)
            ]
            landmark_input = torch.cat(projected, dim=1).flatten(1)
            pose_input = self.global_pool(features[-1]).flatten(1)
            return landmark_input, pose_input

        if self.head_type == "multiscale":
            projected = [
                self.global_pool(block(feature)).flatten(1)
                for block, feature in zip(self.fusion_blocks, features)
            ]
            landmark_input = torch.cat(projected, dim=1)
            pose_input = torch.cat(projected[-2:], dim=1)
            return landmark_input, pose_input

        pooled = self.global_pool(features[-1]).flatten(1)
        return pooled, pooled


    def _heatmaps_to_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, height, width = heatmaps.shape
        logits = heatmaps.view(batch_size, num_points, -1) * self.heatmap_temperature
        probabilities = torch.softmax(logits, dim=-1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=heatmaps.device),
            torch.linspace(0.0, 1.0, width, device=heatmaps.device),
            indexing="ij",
        )
        coord_x = (probabilities * grid_x.reshape(1, 1, -1)).sum(dim=-1)
        coord_y = (probabilities * grid_y.reshape(1, 1, -1)).sum(dim=-1)
        return torch.stack([coord_x, coord_y], dim=-1)


    def _pip_to_coordinates(self, cls_logits: torch.Tensor, x_offset: torch.Tensor, y_offset: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, height, width = cls_logits.shape
        flat_logits = cls_logits.view(batch_size, num_points, -1)
        cell_index = flat_logits.argmax(dim=-1)
        cell_y = torch.div(cell_index, width, rounding_mode="floor")
        cell_x = cell_index % width

        flat_x = x_offset.view(batch_size, num_points, -1)
        flat_y = y_offset.view(batch_size, num_points, -1)
        gathered_x = torch.gather(flat_x, 2, cell_index.unsqueeze(-1)).squeeze(-1).sigmoid()
        gathered_y = torch.gather(flat_y, 2, cell_index.unsqueeze(-1)).squeeze(-1).sigmoid()
        coord_x = (cell_x.to(cls_logits.dtype) + gathered_x) / width
        coord_y = (cell_y.to(cls_logits.dtype) + gathered_y) / height
        return torch.stack([coord_x, coord_y], dim=-1)


    def forward_train(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        features = self.extract_features(inputs)
        if self.head_type == "pip":
            stage2 = features[0]
            refined = self.pip_refine(stage2)
            cls_logits = self.pip_cls_head(refined)
            x_offset = self.pip_x_head(refined)
            y_offset = self.pip_y_head(refined)
            landmarks = self._pip_to_coordinates(cls_logits, x_offset, y_offset)
            pose = self.pose_head(self.global_pool(features[-1]).flatten(1)) if self.pose_head is not None else None
            return landmarks, pose, {"pip_cls": cls_logits, "pip_x": x_offset, "pip_y": y_offset}

        if self.head_type == "deconv_heatmap":
            stage5 = features[-1]
            heatmaps = self.heatmap_head(self.heatmap_refine(stage5))
            landmarks = self._heatmaps_to_coordinates(heatmaps)
            pose = self.pose_head(self.global_pool(stage5).flatten(1)) if self.pose_head is not None else None
            return landmarks, pose, heatmaps

        if self.head_type == "heatmap_hr":
            stage2, stage3, stage4, stage5 = features
            mapped = []
            for block, feature in zip(self.fusion_blocks, (stage2, stage3, stage4, stage5)):
                mapped.append(F.interpolate(block(feature), size=stage2.shape[-2:], mode="bilinear", align_corners=False))
            fused = torch.cat(mapped, dim=1)
            heatmaps = self.heatmap_head(self.heatmap_refine(fused))
            landmarks = self._heatmaps_to_coordinates(heatmaps)
            pose = self.pose_head(self.global_pool(stage5).flatten(1)) if self.pose_head is not None else None
            return landmarks, pose, heatmaps

        if self.head_type == "heatmap":
            stage2, stage3, stage4, stage5 = features
            fused = self.fusion_blocks[0](stage2)
            for block, feature in zip(self.fusion_blocks[1:], (stage3, stage4, stage5)):
                fused = fused + F.interpolate(block(feature), size=fused.shape[-2:], mode="bilinear", align_corners=False)
            heatmaps = self.heatmap_head(self.heatmap_refine(fused))
            landmarks = self._heatmaps_to_coordinates(heatmaps)
            pose = self.pose_head(self.global_pool(stage5).flatten(1)) if self.pose_head is not None else None
            return landmarks, pose, heatmaps

        landmark_input, pose_input = self._pooled_features(features)
        raw_output = self.landmark_head(landmark_input).view(-1, self.num_landmarks, 2)
        if self.residual_mean_shape:
            landmarks = (self.mean_shape.unsqueeze(0) + self.residual_scale * torch.tanh(raw_output)).clamp(0.0, 1.0)
        else:
            landmarks = raw_output.sigmoid()
        pose = self.pose_head(pose_input) if self.pose_head is not None else None
        return landmarks, pose, None


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        landmarks, _, _ = self.forward_train(inputs)
        return landmarks


class HRNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = outputs + residual
        return self.relu(outputs)


def _make_hrnet_branch(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[HRNetBasicBlock(channels, channels) for _ in range(num_blocks)])


def _make_hrnet_transition(prev_channels: list[int], next_channels: list[int]) -> nn.ModuleList:
    transitions: list[nn.Module] = []
    for branch_index, out_channels in enumerate(next_channels):
        if branch_index < len(prev_channels):
            in_channels = prev_channels[branch_index]
            if in_channels == out_channels:
                transitions.append(nn.Identity())
            else:
                transitions.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
            continue

        layers: list[nn.Module] = []
        in_channels = prev_channels[-1]
        downsample_steps = branch_index + 1 - len(prev_channels)
        for step in range(downsample_steps):
            layers.append(ConvBNAct(in_channels, out_channels, 3, stride=2))
            in_channels = out_channels
        transitions.append(nn.Sequential(*layers))
    return nn.ModuleList(transitions)


class HRNetHeatmapNet(nn.Module):
    def __init__(self, *, num_landmarks: int = 68, base_channels: int = 32, num_blocks: int = 2) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_temperature = 8.0
        self.stem = nn.Sequential(
            ConvBNAct(3, base_channels, 3, stride=2),
            ConvBNAct(base_channels, base_channels, 3, stride=2),
            ConvBNAct(base_channels, base_channels * 2, 3),
        )

        stage1_channels = base_channels * 2
        self.stage1 = _make_hrnet_branch(stage1_channels, num_blocks + 1)

        stage2_channels = [stage1_channels, stage1_channels * 2]
        self.transition1 = _make_hrnet_transition([stage1_channels], stage2_channels)
        self.stage2_branches = nn.ModuleList(
            [
                _make_hrnet_branch(stage2_channels[0], num_blocks),
                _make_hrnet_branch(stage2_channels[1], num_blocks),
            ]
        )

        stage3_channels = [stage1_channels, stage1_channels * 2, stage1_channels * 4]
        self.transition2 = _make_hrnet_transition(stage2_channels, stage3_channels)
        self.stage3_branches = nn.ModuleList(
            [
                _make_hrnet_branch(stage3_channels[0], num_blocks),
                _make_hrnet_branch(stage3_channels[1], num_blocks),
                _make_hrnet_branch(stage3_channels[2], num_blocks),
            ]
        )

        stage4_channels = [stage1_channels, stage1_channels * 2, stage1_channels * 4, stage1_channels * 8]
        self.transition3 = _make_hrnet_transition(stage3_channels, stage4_channels)
        self.stage4_branches = nn.ModuleList(
            [
                _make_hrnet_branch(stage4_channels[0], num_blocks),
                _make_hrnet_branch(stage4_channels[1], num_blocks),
                _make_hrnet_branch(stage4_channels[2], num_blocks),
                _make_hrnet_branch(stage4_channels[3], num_blocks),
            ]
        )
        self.final_projections = nn.ModuleList(
            [
                nn.Identity(),
                ConvBNAct(stage4_channels[1], stage4_channels[0], 1),
                ConvBNAct(stage4_channels[2], stage4_channels[0], 1),
                ConvBNAct(stage4_channels[3], stage4_channels[0], 1),
            ]
        )

        self.head = nn.Sequential(
            ConvBNAct(stage4_channels[0], stage4_channels[0], 3),
            nn.Conv2d(stage4_channels[0], num_landmarks, 1),
        )


    def _heatmaps_to_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, height, width = heatmaps.shape
        logits = heatmaps.view(batch_size, num_points, -1) * self.heatmap_temperature
        probabilities = torch.softmax(logits, dim=-1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=heatmaps.device),
            torch.linspace(0.0, 1.0, width, device=heatmaps.device),
            indexing="ij",
        )
        coord_x = (probabilities * grid_x.reshape(1, 1, -1)).sum(dim=-1)
        coord_y = (probabilities * grid_y.reshape(1, 1, -1)).sum(dim=-1)
        return torch.stack([coord_x, coord_y], dim=-1)


    def _fuse_to_high_resolution(self, branches: list[torch.Tensor]) -> torch.Tensor:
        target = branches[0]
        fused = branches[0]
        for branch in branches[1:]:
            projected = branch
            if projected.shape[-2:] != target.shape[-2:]:
                projected = F.interpolate(projected, size=target.shape[-2:], mode="bilinear", align_corners=False)
            fused = fused + projected
        return fused


    def forward_train(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor]:
        stage0 = self.stem(inputs)
        stage1 = self.stage1(stage0)

        branch1 = self.transition1[0](stage1)
        branch2 = self.transition1[1](stage1)
        branch1 = self.stage2_branches[0](branch1)
        branch2 = self.stage2_branches[1](branch2)

        branch3_input = [branch1, branch2]
        next_stage3 = []
        for branch_index, transition in enumerate(self.transition2):
            if branch_index < len(branch3_input):
                next_stage3.append(transition(branch3_input[branch_index]))
            else:
                next_stage3.append(transition(branch3_input[-1]))
        stage3 = [module(feature) for module, feature in zip(self.stage3_branches, next_stage3)]

        branch4_input = stage3
        next_stage4 = []
        for branch_index, transition in enumerate(self.transition3):
            if branch_index < len(branch4_input):
                next_stage4.append(transition(branch4_input[branch_index]))
            else:
                next_stage4.append(transition(branch4_input[-1]))
        stage4 = [module(feature) for module, feature in zip(self.stage4_branches, next_stage4)]

        fused = self.final_projections[0](stage4[0])
        for projection, branch in zip(self.final_projections[1:], stage4[1:]):
            fused = fused + F.interpolate(projection(branch), size=fused.shape[-2:], mode="bilinear", align_corners=False)
        heatmaps = self.head(fused)
        landmarks = self._heatmaps_to_coordinates(heatmaps)
        return landmarks, None, heatmaps


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        landmarks, _, _ = self.forward_train(inputs)
        return landmarks


class PFLDInvertedResidual(nn.Module):
    """PFLD-style inverted residual: pw expand → dw 3x3 → pw project, residual when stride=1 and channels match."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden, 1))
        layers.append(ConvBNAct(hidden, hidden, 3, stride=stride, groups=hidden))
        layers.extend([nn.Conv2d(hidden, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)])
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.block(inputs)
        if self.use_residual:
            outputs = outputs + inputs
        return outputs


class PFLDNet(nn.Module):
    """PFLD-style backbone for face landmark regression.

    Reference: Guo et al., "PFLD: A Practical Facial Landmark Detector" (2019).
    Adapted for 256x256 input (one extra stride-2 stage compared to original 112x112 design).
    Optional PIP head (cls grid + sub-pixel offset) replaces the FC regression head.
    Optional auxiliary pose head used during training for regularization.
    """

    def __init__(
        self,
        *,
        num_landmarks: int = 106,
        width_mult: float = 1.0,
        head_type: str = "fc",
        pip_grid: int = 8,
        aux_pose_head: bool = True,
        roi_size: int = 4,
        roi_channels: int = 16,
        mlp_hidden: int = 64,
        refinement_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.head_type = head_type
        self.pip_grid = pip_grid
        self.use_pose_head = aux_pose_head
        self.roi_size = roi_size
        self.roi_channels = roi_channels
        self.mlp_hidden = mlp_hidden

        c = lambda n: _round_channels(n, width_mult)
        # Stage 1+2: 256→128→128 (stride 2 then stride 1)
        self.stage1 = ConvBNAct(3, c(64), 3, stride=2)
        self.stage2 = ConvBNAct(c(64), c(64), 3, stride=1, groups=c(64))
        # Stage 3: 5x InvBlock e=2, [2,1,1,1,1] → 64 (256/4)
        s3_out = c(64)
        self.stage3 = nn.Sequential(
            PFLDInvertedResidual(c(64), s3_out, stride=2, expand_ratio=2),
            PFLDInvertedResidual(s3_out, s3_out, stride=1, expand_ratio=2),
            PFLDInvertedResidual(s3_out, s3_out, stride=1, expand_ratio=2),
            PFLDInvertedResidual(s3_out, s3_out, stride=1, expand_ratio=2),
            PFLDInvertedResidual(s3_out, s3_out, stride=1, expand_ratio=2),
        )
        # Stage 4: 1x InvBlock e=2 stride=2 → 32 channels
        s4_out = c(128)
        self.stage4 = PFLDInvertedResidual(s3_out, s4_out, stride=2, expand_ratio=2)
        # Stage 5: 6x InvBlock e=4 stride=1 → 32, c=128
        self.stage5 = nn.Sequential(
            *[PFLDInvertedResidual(s4_out, s4_out, stride=1, expand_ratio=4) for _ in range(6)]
        )
        # Stage 6: 1x InvBlock e=2 stride=2 → 16, c=16 (PFLD bottleneck)
        s6_out = c(16)
        self.stage6 = PFLDInvertedResidual(s4_out, s6_out, stride=2, expand_ratio=2)
        # Stage 7: Conv 3x3 stride=2 → 8, c=32
        s7_out = c(32)
        self.stage7 = ConvBNAct(s6_out, s7_out, 3, stride=2)
        # Stage 8: Conv on 8x8 → 1x1, c=128
        s8_out = c(128)
        self.stage8 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(s7_out, s8_out, 1, bias=False),
            nn.BatchNorm2d(s8_out),
            nn.ReLU(inplace=True),
        )

        # multi-scale fusion: GAP each scale → concat
        self.gap = nn.AdaptiveAvgPool2d(1)
        fused_dim = s6_out + s7_out + s8_out

        if head_type in {"pip", "pip_cascade"}:
            # PIP head on stage5 output (32x32). Coarse grid pip_grid x pip_grid via pooling.
            self.pip_pool = nn.AdaptiveAvgPool2d(pip_grid)
            self.pip_proj = ConvBNAct(s4_out, s4_out, 1)
            self.pip_cls_head = nn.Conv2d(s4_out, num_landmarks, 1)
            self.pip_x_head = nn.Conv2d(s4_out, num_landmarks, 1)
            self.pip_y_head = nn.Conv2d(s4_out, num_landmarks, 1)
            if head_type == "pip_cascade":
                self.roi_proj = nn.Conv2d(s4_out, roi_channels, 1)
                roi_feat_dim = roi_channels * roi_size * roi_size
                self.refinement_mlp = nn.Sequential(
                    nn.Linear(roi_feat_dim, mlp_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(mlp_hidden, mlp_hidden // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(mlp_hidden // 2, 2),
                )
                if refinement_scale is None:
                    refinement_scale = float(roi_size) / 32.0
                self.refinement_range = refinement_scale
                last = self.refinement_mlp[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            else:
                self.roi_proj = None
                self.refinement_mlp = None
                self.refinement_range = None
            self.landmark_head = None
        else:
            self.register_buffer("mean_shape", torch.full((num_landmarks, 2), 0.5), persistent=False)
            self.landmark_head = nn.Sequential(
                nn.Linear(fused_dim, num_landmarks * 2),
            )
            self.pip_pool = None
            self.roi_proj = None
            self.refinement_mlp = None
            self.refinement_range = None

        if aux_pose_head:
            # Auxiliary pose net (used only during training): predicts roll/yaw/pitch from stage3
            self.aux_pose = nn.Sequential(
                ConvBNAct(s3_out, c(128), 3, stride=2),
                ConvBNAct(c(128), c(128), 3, stride=1),
                ConvBNAct(c(128), c(32), 3, stride=2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c(32), 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 3),
            )
        else:
            self.aux_pose = None

    def initialize_landmark_bias(self, mean_shape: torch.Tensor) -> None:
        if self.landmark_head is None:
            return
        with torch.no_grad():
            bias = mean_shape.flatten() * 4.0 - 2.0  # inverse sigmoid of mean_shape ∈ [0,1]
            final_linear = self.landmark_head[-1]
            final_linear.bias.copy_(bias)
            self.mean_shape.copy_(mean_shape)

    def _pip_to_coordinates(self, cls_logits, x_offset, y_offset):
        b, n, h, w = cls_logits.shape
        flat = cls_logits.view(b, n, -1)
        cell = flat.argmax(dim=-1)
        cell_y = torch.div(cell, w, rounding_mode="floor")
        cell_x = cell % w
        flat_x = x_offset.view(b, n, -1)
        flat_y = y_offset.view(b, n, -1)
        gx = torch.gather(flat_x, 2, cell.unsqueeze(-1)).squeeze(-1).sigmoid()
        gy = torch.gather(flat_y, 2, cell.unsqueeze(-1)).squeeze(-1).sigmoid()
        coord_x = (cell_x.to(cls_logits.dtype) + gx) / w
        coord_y = (cell_y.to(cls_logits.dtype) + gy) / h
        return torch.stack([coord_x, coord_y], dim=-1)

    def _roi_refine(self, feature_map: torch.Tensor, coarse_coords: torch.Tensor) -> torch.Tensor:
        batch, num_points, _ = coarse_coords.shape
        compact = self.roi_proj(feature_map)
        feature_size = float(feature_map.shape[-1])
        radius_norm = float(self.roi_size) / (2.0 * feature_size)
        device = coarse_coords.device
        offsets_1d = torch.linspace(-radius_norm, radius_norm, self.roi_size, device=device)
        gy, gx = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        local_grid = torch.stack([gx, gy], dim=-1)

        center = coarse_coords.unsqueeze(2).unsqueeze(2)
        sample_xy = (center + local_grid[None, None, :, :, :]) * 2.0 - 1.0
        sample_grid = sample_xy.reshape(batch, num_points * self.roi_size, self.roi_size, 2)

        sampled = F.grid_sample(compact, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.view(batch, self.roi_channels, num_points, self.roi_size, self.roi_size)
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()

        flat = sampled.reshape(batch * num_points, -1)
        delta = self.refinement_mlp(flat)
        delta = torch.tanh(delta) * self.refinement_range
        return delta.view(batch, num_points, 2)

    def forward_train(self, inputs):
        x = self.stage1(inputs)
        x = self.stage2(x)
        s3 = self.stage3(x)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        s6 = self.stage6(s5)
        s7 = self.stage7(s6)
        s8 = self.stage8(s7)

        pose = self.aux_pose(s3) if self.aux_pose is not None else None

        if self.head_type in {"pip", "pip_cascade"}:
            pooled = self.pip_pool(s5)
            refined = self.pip_proj(pooled)
            cls_logits = self.pip_cls_head(refined)
            x_off = self.pip_x_head(refined)
            y_off = self.pip_y_head(refined)
            landmarks = self._pip_to_coordinates(cls_logits, x_off, y_off)
            if self.head_type == "pip_cascade":
                delta = self._roi_refine(s5, landmarks)
                landmarks = (landmarks + delta).clamp(0.0, 1.0)
            return landmarks, pose, {"pip_cls": cls_logits, "pip_x": x_off, "pip_y": y_off}

        # FC head: multi-scale GAP fusion
        f1 = self.gap(s6).flatten(1)
        f2 = self.gap(s7).flatten(1)
        f3 = s8.flatten(1)
        fused = torch.cat([f1, f2, f3], dim=1)
        landmarks = self.landmark_head(fused).view(-1, self.num_landmarks, 2).sigmoid()
        return landmarks, pose, None

    def forward(self, inputs):
        landmarks, _, _ = self.forward_train(inputs)
        return landmarks


class MNV3InvertedResidual(nn.Module):
    """MobileNetV3-style inverted residual: pw expand → dw 3x3 → SE → pw project."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: int, use_se: bool = True) -> None:
        super().__init__()
        hidden = in_ch * expand
        self.use_residual = stride == 1 and in_ch == out_ch
        layers: list[nn.Module] = []
        if expand != 1:
            layers.append(ConvBNAct(in_ch, hidden, 1))
        layers.append(ConvBNAct(hidden, hidden, 3, stride=stride, groups=hidden))
        if use_se:
            layers.append(SqueezeExcitation(hidden))
        layers.extend([nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MRFFNNet(nn.Module):
    """Multi-Resolution Feature-pyramid + Face-shape Network.

    Components:
    1. MobileNetV3-style backbone producing 4 scales: F1(64x64), F2(32x32), F3(16x16), F4(8x8)
    2. FPN top-down lateral fusion to D=32 channels at each scale
    3. Coarse heatmap head on P2 (32x32, upsampled to 64x64 via the prior pyramid level)
    4. PCA Shape Prior Module: predict K mixing coefficients from global feature
       and combine with mean shape + per-point residual

    Designed for INT8 ≤ 2 MB (ICME 2021 limit). Predicted ~1M params.
    """

    def __init__(
        self,
        *,
        num_landmarks: int = 106,
        width_mult: float = 1.0,
        fpn_dim: int = 32,
        heatmap_size: int = 64,
        pca_basis: int = 8,
        use_pca: bool = True,
        pca_npz_path: str | None = None,
        heatmap_temperature: float = 8.0,
    ) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        self.heatmap_temperature = heatmap_temperature
        self.use_pca = use_pca
        self._num_pca_basis = pca_basis

        c = lambda n: _round_channels(n, width_mult)
        # Stem: 256 → 128, 16 ch
        self.stem = nn.Sequential(
            ConvBNAct(3, c(16), 3, stride=2),
            MNV3InvertedResidual(c(16), c(16), stride=1, expand=1, use_se=False),
        )
        # Stage 1: 128 → 64, c(24)  (F1)
        self.stage1 = nn.Sequential(
            MNV3InvertedResidual(c(16), c(24), stride=2, expand=4, use_se=False),
            MNV3InvertedResidual(c(24), c(24), stride=1, expand=3, use_se=False),
        )
        # Stage 2: 64 → 32, c(40)  (F2)
        self.stage2 = nn.Sequential(
            MNV3InvertedResidual(c(24), c(40), stride=2, expand=4, use_se=True),
            MNV3InvertedResidual(c(40), c(40), stride=1, expand=4, use_se=True),
            MNV3InvertedResidual(c(40), c(40), stride=1, expand=4, use_se=True),
        )
        # Stage 3: 32 → 16, c(80)  (F3)
        self.stage3 = nn.Sequential(
            MNV3InvertedResidual(c(40), c(80), stride=2, expand=6, use_se=True),
            MNV3InvertedResidual(c(80), c(80), stride=1, expand=6, use_se=True),
            MNV3InvertedResidual(c(80), c(80), stride=1, expand=6, use_se=True),
            MNV3InvertedResidual(c(80), c(80), stride=1, expand=6, use_se=True),
        )
        # Stage 4: 16 → 8, c(128) (F4)
        self.stage4 = nn.Sequential(
            MNV3InvertedResidual(c(80), c(128), stride=2, expand=6, use_se=True),
            MNV3InvertedResidual(c(128), c(128), stride=1, expand=6, use_se=True),
        )

        # FPN lateral 1x1 + smooth 3x3 (top-down)
        self.lateral_f1 = nn.Conv2d(c(24), fpn_dim, 1)
        self.lateral_f2 = nn.Conv2d(c(40), fpn_dim, 1)
        self.lateral_f3 = nn.Conv2d(c(80), fpn_dim, 1)
        self.lateral_f4 = nn.Conv2d(c(128), fpn_dim, 1)
        self.smooth_p1 = ConvBNAct(fpn_dim, fpn_dim, 3)
        self.smooth_p2 = ConvBNAct(fpn_dim, fpn_dim, 3)
        self.smooth_p3 = ConvBNAct(fpn_dim, fpn_dim, 3)

        # Coarse heatmap head on P1 (64x64)
        self.coarse_head = nn.Conv2d(fpn_dim, num_landmarks, 1)

        # PCA Shape Prior Module
        if use_pca:
            self.pca_global_pool = nn.AdaptiveAvgPool2d(1)
            self.pca_coef_head = nn.Sequential(
                nn.Linear(c(128), 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, pca_basis),
            )
            # Residual MLP from coarse coords
            self.pca_residual_head = nn.Sequential(
                nn.Linear(num_landmarks * 2, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_landmarks * 2),
            )
            # Buffers (filled at init / via load_pca)
            self.register_buffer("pca_mean_shape", torch.full((num_landmarks, 2), 0.5))
            self.register_buffer("pca_basis", torch.zeros(pca_basis, num_landmarks, 2))
            # Learnable scaling for residual blend (start small)
            self.residual_scale = nn.Parameter(torch.tensor(0.1))
            if pca_npz_path is not None:
                self.load_pca(pca_npz_path)

    def load_pca(self, npz_path: str) -> None:
        import numpy as np
        data = np.load(npz_path)
        mean = torch.from_numpy(data["mean_shape"]).float()  # (106, 2)
        basis = torch.from_numpy(data["basis"]).float()  # (K, 106, 2)
        if basis.shape[0] >= self._num_pca_basis:
            basis = basis[: self._num_pca_basis]
        else:
            raise ValueError(
                f"PCA file has only {basis.shape[0]} basis vectors, need {self._num_pca_basis}"
            )
        self.pca_mean_shape.copy_(mean)
        self.pca_basis.copy_(basis)
        print(f"[MRFFNNet] loaded PCA: mean_shape range=[{mean.min():.3f}, {mean.max():.3f}], "
              f"basis shape={tuple(basis.shape)}")

    def _heatmap_to_coords(self, heatmap: torch.Tensor) -> torch.Tensor:
        b, n, h, w = heatmap.shape
        logits = heatmap.view(b, n, -1) * self.heatmap_temperature
        prob = torch.softmax(logits, dim=-1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=heatmap.device),
            torch.linspace(0.0, 1.0, w, device=heatmap.device),
            indexing="ij",
        )
        cx = (prob * grid_x.reshape(1, 1, -1)).sum(dim=-1)
        cy = (prob * grid_y.reshape(1, 1, -1)).sum(dim=-1)
        return torch.stack([cx, cy], dim=-1)

    def forward_train(self, inputs: torch.Tensor):
        x = self.stem(inputs)
        f1 = self.stage1(x)   # 64x64
        f2 = self.stage2(f1)  # 32x32
        f3 = self.stage3(f2)  # 16x16
        f4 = self.stage4(f3)  # 8x8

        # FPN top-down
        p4 = self.lateral_f4(f4)
        p3 = self.smooth_p3(self.lateral_f3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="nearest"))
        p2 = self.smooth_p2(self.lateral_f2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="nearest"))
        p1 = self.smooth_p1(self.lateral_f1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="nearest"))

        # Coarse heatmap on P1 (64x64) — matches HRNet teacher output size
        heatmap = self.coarse_head(p1)
        coarse_coords = self._heatmap_to_coords(heatmap)

        if not self.use_pca:
            return coarse_coords, None, heatmap

        # PCA module: predict mixing coefs from global feature, residual from coarse coords
        global_feat = self.pca_global_pool(f4).flatten(1)  # (B, c(128))
        coefs = self.pca_coef_head(global_feat)  # (B, K)
        # Reconstruct shape from PCA: mean + sum(coef * basis)
        # mean_shape: (106, 2), basis: (K, 106, 2), coefs: (B, K)
        flat_basis = self.pca_basis.view(self._num_pca_basis, -1)  # (K, 212)
        flat_mean = self.pca_mean_shape.view(-1)  # (212,)
        pca_recon = flat_mean[None, :] + coefs @ flat_basis  # (B, 212)
        pca_recon = pca_recon.view(-1, self.num_landmarks, 2)

        # Residual from coarse coords
        residual = self.pca_residual_head(coarse_coords.flatten(1))  # (B, 212)
        residual = residual.view(-1, self.num_landmarks, 2)

        # Combine: PCA reconstruction + residual scaling × coord-residual
        landmarks = pca_recon + self.residual_scale * residual
        landmarks = landmarks.clamp(0.0, 1.0)
        return landmarks, None, heatmap

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        landmarks, _, _ = self.forward_train(inputs)
        return landmarks


class CSPRNet(MRFFNNet):
    """Cascade Sub-Pixel Refinement Network.

    Stage 1: MR-FFN backbone -> coarse landmarks (PCA-blended)
    Stage 2: Per-point ROI sampling on P1 features -> shared MLP -> sub-pixel offset

    Designed to push 1.5M-class models beyond ~5% NME plateau without exceeding 2 MB
    INT8. Sub-pixel refinement targets the tail (10-15%) of points where NME is high.
    """

    def __init__(
        self,
        *,
        roi_size: int = 4,
        roi_channels: int = 16,
        mlp_hidden: int = 64,
        refinement_scale: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.roi_size = roi_size
        self.roi_channels = roi_channels
        self.mlp_hidden = mlp_hidden

        fpn_dim = self.lateral_f1.out_channels
        self.roi_proj = nn.Conv2d(fpn_dim, roi_channels, 1)

        roi_feat_dim = roi_channels * roi_size * roi_size
        self.refinement_mlp = nn.Sequential(
            nn.Linear(roi_feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden // 2, 2),
        )
        # Refinement range in normalized [0,1] coords. Default = roi_size / heatmap_size.
        if refinement_scale is None:
            refinement_scale = float(roi_size) / float(self.heatmap_size)
        self.refinement_range = refinement_scale

        # zero-init last layer so refined ≈ stage1 at training start
        last = self.refinement_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _stage1_forward(self, inputs: torch.Tensor):
        x = self.stem(inputs)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        p4 = self.lateral_f4(f4)
        p3 = self.smooth_p3(self.lateral_f3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="nearest"))
        p2 = self.smooth_p2(self.lateral_f2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="nearest"))
        p1 = self.smooth_p1(self.lateral_f1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="nearest"))

        heatmap = self.coarse_head(p1)
        coarse_coords = self._heatmap_to_coords(heatmap)

        if self.use_pca:
            global_feat = self.pca_global_pool(f4).flatten(1)
            coefs = self.pca_coef_head(global_feat)
            flat_basis = self.pca_basis.view(self._num_pca_basis, -1)
            flat_mean = self.pca_mean_shape.view(-1)
            pca_recon = (flat_mean[None, :] + coefs @ flat_basis).view(-1, self.num_landmarks, 2)
            residual = self.pca_residual_head(coarse_coords.flatten(1)).view(-1, self.num_landmarks, 2)
            stage1_coords = (pca_recon + self.residual_scale * residual).clamp(0.0, 1.0)
        else:
            stage1_coords = coarse_coords

        return stage1_coords, heatmap, p1

    def _roi_refine(self, p1: torch.Tensor, stage1_coords: torch.Tensor) -> torch.Tensor:
        batch, num_points, _ = stage1_coords.shape
        compact = self.roi_proj(p1)

        # Build per-point sampling grid
        # grid_sample expects coords in [-1, 1]; we sample a (roi_size x roi_size) patch
        # centered at each landmark. Patch radius in [0,1] = roi_size / (2 * heatmap_size).
        radius_norm = float(self.roi_size) / (2.0 * float(self.heatmap_size))
        device = stage1_coords.device
        offsets_1d = torch.linspace(-radius_norm, radius_norm, self.roi_size, device=device)
        gy, gx = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        local_grid = torch.stack([gx, gy], dim=-1)  # (roi, roi, 2) in [0,1] space

        center = stage1_coords.unsqueeze(2).unsqueeze(2)  # (B, N, 1, 1, 2)
        sample_xy = (center + local_grid[None, None, :, :, :]) * 2.0 - 1.0  # to [-1, 1]
        # flatten last two dims into (H_out=N*roi, W_out=roi)
        sample_grid = sample_xy.reshape(batch, num_points * self.roi_size, self.roi_size, 2)

        sampled = F.grid_sample(compact, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
        # sampled: (B, roi_channels, N*roi, roi) -> (B, N, roi_channels, roi, roi)
        sampled = sampled.view(batch, self.roi_channels, num_points, self.roi_size, self.roi_size)
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()

        flat = sampled.reshape(batch * num_points, -1)
        delta = self.refinement_mlp(flat)
        delta = torch.tanh(delta) * self.refinement_range
        return delta.view(batch, num_points, 2)

    def forward_train(self, inputs: torch.Tensor):
        stage1_coords, heatmap, p1 = self._stage1_forward(inputs)
        delta = self._roi_refine(p1, stage1_coords)
        refined = (stage1_coords + delta).clamp(0.0, 1.0)
        return refined, None, heatmap

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        landmarks, _, _ = self.forward_train(inputs)
        return landmarks



def build_model(config: dict) -> LMNet:
    model_config = config["model"]
    architecture = str(model_config.get("arch", "lmnet"))
    if architecture == "hrnet_w18":
        return HRNetHeatmapNet(
            num_landmarks=int(config["data"]["num_landmarks"]),
            base_channels=int(model_config.get("base_channels", 32)),
            num_blocks=int(model_config.get("num_blocks", 2)),
        )
    if architecture == "pfld":
        return PFLDNet(
            num_landmarks=int(config["data"]["num_landmarks"]),
            width_mult=float(model_config.get("width_mult", 1.0)),
            head_type=str(model_config.get("head_type", "fc")),
            pip_grid=int(model_config.get("pip_grid", 8)),
            aux_pose_head=bool(model_config.get("pose_head", True)),
            roi_size=int(model_config.get("roi_size", 4)),
            roi_channels=int(model_config.get("roi_channels", 16)),
            mlp_hidden=int(model_config.get("mlp_hidden", 64)),
            refinement_scale=model_config.get("refinement_scale"),
        )
    if architecture == "mrffn":
        return MRFFNNet(
            num_landmarks=int(config["data"]["num_landmarks"]),
            width_mult=float(model_config.get("width_mult", 1.0)),
            fpn_dim=int(model_config.get("fpn_dim", 32)),
            heatmap_size=int(model_config.get("heatmap_size", 64)),
            pca_basis=int(model_config.get("pca_basis", 8)),
            use_pca=bool(model_config.get("use_pca", True)),
            pca_npz_path=model_config.get("pca_npz_path"),
            heatmap_temperature=float(model_config.get("heatmap_temperature", 8.0)),
        )
    if architecture == "csprnet":
        return CSPRNet(
            num_landmarks=int(config["data"]["num_landmarks"]),
            width_mult=float(model_config.get("width_mult", 1.0)),
            fpn_dim=int(model_config.get("fpn_dim", 32)),
            heatmap_size=int(model_config.get("heatmap_size", 64)),
            pca_basis=int(model_config.get("pca_basis", 8)),
            use_pca=bool(model_config.get("use_pca", True)),
            pca_npz_path=model_config.get("pca_npz_path"),
            heatmap_temperature=float(model_config.get("heatmap_temperature", 8.0)),
            roi_size=int(model_config.get("roi_size", 4)),
            roi_channels=int(model_config.get("roi_channels", 16)),
            mlp_hidden=int(model_config.get("mlp_hidden", 64)),
            refinement_scale=model_config.get("refinement_scale"),
        )
    if architecture == "resnet18_deconv":
        return ResNetDeconvHeatmapNet(
            num_landmarks=int(config["data"]["num_landmarks"]),
            pretrained_backbone=bool(model_config.get("pretrained_backbone", True)),
            deconv_dim=int(model_config.get("deconv_dim", 256)),
        )
    if architecture == "fan_heatmap":
        return FANHeatmapNet(
            num_modules=int(model_config.get("num_modules", 2)),
            num_landmarks=int(config["data"]["num_landmarks"]),
        )
    if architecture == "unet_heatmap":
        return UNetHeatmapNet(
            base_channels=int(model_config.get("base_channels", 48)),
            num_landmarks=int(config["data"]["num_landmarks"]),
        )
    return LMNet(
        width_mult=float(model_config["width_mult"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        use_se=bool(model_config["use_se"]),
        num_landmarks=int(config["data"]["num_landmarks"]),
        head_type=str(model_config.get("head_type", "multiscale")),
        fusion_dim=int(model_config.get("fusion_dim", 160)),
        pose_head=bool(model_config.get("pose_head", False)),
        residual_mean_shape=bool(model_config.get("residual_mean_shape", False)),
        residual_scale=float(model_config.get("residual_scale", 0.25)),
    )
