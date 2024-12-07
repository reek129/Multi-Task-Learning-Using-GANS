# model.py
import torch
import torch.nn as nn


# Define the CRPBlock
class CRPBlock(nn.Module):
    def __init__(self, in_chans, out_chans, n_stages=4, groups=False):
        super(CRPBlock, self).__init__()
        self.n_stages = n_stages
        groups = in_chans if groups else 1
        self.mini_blocks = nn.ModuleList()
        for _ in range(n_stages):
            self.mini_blocks.append(nn.MaxPool2d(kernel_size=5, stride=1, padding=2))
            self.mini_blocks.append(nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False, groups=groups))
    
    def forward(self, x):
        out = x
        for block in self.mini_blocks:
            out = block(out)
            x = x + out
        return x

class MobileNetV3Backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.proj_l1 = nn.Conv2d(16, 576, kernel_size=1, bias=False)   # For l1_out (1/4 resolution)
        self.proj_l3 = nn.Conv2d(24, 576, kernel_size=1, bias=False)  # For l3_out (1/8 resolution)
        self.proj_l7 = nn.Conv2d(48, 576, kernel_size=1, bias=False)  # For l7_out (1/16 resolution)
        self.proj_l11 = nn.Conv2d(96, 576, kernel_size=1, bias=False) # For l11_out (1/32 resolution)

    
    def forward(self, x):
        """ Passes input theough MobileNetV3 backbone feature extraction layers
            layers to add connections to (0 indexed)
                - 1:  1/4 res
                - 3:  1/8 res
                - 7, 8:  1/16 res
                - 10, 11: 1/32 res
           """
        # skips = nn.ParameterDict()
        # for i in range(len(self.backbone) - 1):
        #     x = self.backbone[i](x)
        #     # add skip connection outputs
        #     if i in [1, 3, 7, 11]:
        #         skips.update({f"l{i}_out" : x})

        # return skips
        skips = {}  # Dictionary to store skip connections

        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # Add skip connections for specific layers
            if i == 1:
                skips["l1_out"] = self.proj_l1(x)  # Project l1_out
            elif i == 3:
                skips["l3_out"] = self.proj_l3(x)  # Project l3_out
            elif i == 7:
                skips["l7_out"] = self.proj_l7(x)  # Project l7_out
            elif i == 11:
                skips["l11_out"] = self.proj_l11(x)  # Project l11_out

        return skips

class EnhancedSharedGenerator(nn.Module):
    def __init__(self):
        """
        Enhanced Shared Generator for Segmentation and Depth tasks.
        Includes additional refinement layers for better generalization.
        """
        super(EnhancedSharedGenerator, self).__init__()
        # Shared convolution layers to process each skip connection
        self.shared_conv1 = nn.Conv2d(576, 256, kernel_size=1, bias=False)  # Process l11_out (1/32)
        self.shared_conv2 = nn.Conv2d(576, 256, kernel_size=1, bias=False)  # Process l7_out (1/16)
        self.shared_conv3 = nn.Conv2d(576, 256, kernel_size=1, bias=False)  # Process l3_out (1/8)
        self.shared_conv4 = nn.Conv2d(576, 256, kernel_size=1, bias=False)  # Process l1_out (1/4)

        # CRP blocks for refinement
        self.shared_crp1 = CRPBlock(256, 256, n_stages=4)
        self.shared_crp2 = CRPBlock(256, 256, n_stages=4)
        self.shared_crp3 = CRPBlock(256, 256, n_stages=4)
        self.shared_crp4 = CRPBlock(256, 256, n_stages=4)

        # Additional refinement layers
        self.refine = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

    def forward(self, skips):
        """
        Process skips with shared layers for task-specific generation.
        Args:
            skips (dict): Skip connections from the encoder.
        Returns:
            dict: Processed skip connections.
        """
        x1 = self.shared_crp1(self.shared_conv1(skips["l11_out"]))
        x1 = self.refine(x1)  # Extra refinement

        x2 = self.shared_crp2(self.shared_conv2(skips["l7_out"]))
        x2 = self.refine(x2)

        x3 = self.shared_crp3(self.shared_conv3(skips["l3_out"]))
        x3 = self.refine(x3)

        x4 = self.shared_crp4(self.shared_conv4(skips["l1_out"]))
        x4 = self.refine(x4)

        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4}


class TaskOutputLayer(nn.Module):
    def __init__(self, output_channels):
        """
        Task-specific output layers for generating final predictions.
        Args:
            output_channels (int): Number of output channels (e.g., 20 for segmentation, 1 for depth).
        """
        super(TaskOutputLayer, self).__init__()
        self.final_conv = nn.Conv2d(256, output_channels, kernel_size=3, padding=1)

    def forward(self, x, input_size):
        """
        Generate task-specific output.
        Args:
            x (Tensor): Input feature map.
            input_size (tuple): Original input size (H, W).
        Returns:
            Tensor: Task-specific output.
        """
        x = self.final_conv(x)
        return nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


class TaskSpecificDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(TaskSpecificDiscriminator, self).__init__()
        self.adapt_conv = nn.Conv2d(input_channels+input_channels, input_channels, kernel_size=1, bias=False)
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, task_output, labels=None):
        """
        Forward pass through the discriminator.

        Args:
            task_output (Tensor): Output from the generator (e.g., seg_output or depth_output).
            labels (Tensor, optional): Ground truth labels. If provided, aligns channels with task_output.

        Returns:
            Tensor: Discriminator's prediction.
        """
        if labels is not None:
            # Ensure labels match the shape of task_output
            if labels.dim() < task_output.dim():
                labels = labels.unsqueeze(1)  # Add channel dimension if needed
            if labels.size(1) != task_output.size(1):
                labels = torch.nn.functional.one_hot(labels.squeeze(1), num_classes=task_output.size(1))
                labels = labels.permute(0, 3, 1, 2).float().to(task_output.device)
            combined = torch.cat([task_output, labels], dim=1)
            combined = self.adapt_conv(combined)
        else:
            combined = task_output

        return self.model(combined)


class MultiTaskDiscriminator(nn.Module):
    def __init__(self, input_channels):
        """
        Multi-Task Discriminator for evaluating all task-specific outputs.
        Args:
            input_channels (int): Number of input channels for concatenated features and outputs.
        """
        super(MultiTaskDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, inputs):
        """
        Evaluate input image and task-specific outputs.
        Args:
            inputs (Tensor): Input image.
            outputs (list[Tensor]): List of task-specific outputs.
        Returns:
            Tensor: Discriminator output.
        """
        # combined = torch.cat([inputs] + outputs, dim=1)
        return self.model(inputs)


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_seg_classes=20, depth_channels=1):
        """
        Multi-task model with shared Pix2Pix Generator, task-specific discriminators,
        and a multi-task discriminator.
        Args:
            backbone (nn.Module): Encoder backbone for feature extraction.
            num_seg_classes (int): Number of segmentation classes.
            depth_channels (int): Number of output channels for depth.
        """
        super(MultiTaskModel, self).__init__()
        self.feature_generator = MobileNetV3Backbone(backbone)
        self.shared_generator = EnhancedSharedGenerator()
        self.seg_output_layer = TaskOutputLayer(output_channels=num_seg_classes)
        self.depth_output_layer = TaskOutputLayer(output_channels=depth_channels)

        # Task-specific discriminators
        self.seg_discriminator = TaskSpecificDiscriminator(input_channels=num_seg_classes)
        self.depth_discriminator = TaskSpecificDiscriminator(input_channels=depth_channels)

        # Multi-task discriminator
        self.multi_task_discriminator = MultiTaskDiscriminator(input_channels=3 + num_seg_classes + depth_channels)
        
    def forward(self, inputs, input_size, seg_labels=None, depth_labels=None, return_discriminator_outputs=False):
        # Extract features from the encoder
        skips = self.feature_generator(inputs)
        shared_features = self.shared_generator(skips)

        # Task-specific outputs
        seg_output = self.seg_output_layer(shared_features["x4"], input_size)
        depth_output = self.depth_output_layer(shared_features["x4"], input_size)

        output_dict = {
            "seg_output": seg_output,
            "depth_output": depth_output,
        }

        if return_discriminator_outputs:
            
            # Detach outputs to prevent discriminator backward from interfering with the generator
            seg_output_detached = seg_output.detach()
            depth_output_detached = depth_output.detach()
            
            # Adversarial feedback from task-specific discriminators
            seg_real_disc = self.seg_discriminator(seg_output_detached, seg_labels) if seg_labels is not None else None
            seg_fake_disc = self.seg_discriminator(seg_output_detached, None)

            depth_real_disc = self.depth_discriminator(depth_output_detached, depth_labels) if depth_labels is not None else None
            depth_fake_disc = self.depth_discriminator(depth_output_detached, None)

            # Multi-task discriminator feedback
            combined_real_input = torch.cat([inputs, seg_labels, depth_labels], dim=1) if seg_labels is not None and depth_labels is not None else None
            combined_fake_input = torch.cat([inputs, seg_output, depth_output], dim=1)

            combined_real_disc = self.multi_task_discriminator(combined_real_input) if combined_real_input is not None else None
            combined_fake_disc = self.multi_task_discriminator(combined_fake_input.detach())

            output_dict.update({
                "seg_real_disc": seg_real_disc,
                "seg_fake_disc": seg_fake_disc,
                "depth_real_disc": depth_real_disc,
                "depth_fake_disc": depth_fake_disc,
                "combined_real_disc": combined_real_disc,
                "combined_fake_disc": combined_fake_disc,
            })

        return output_dict

