from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch import nn
import torch

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from monai.networks.nets import SegResNetVAE
from torch.optim import Adam

'''
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 603 3d_fullres all -device cuda --npz -tr nnUNetTrainerSegResNetVAE
'''


import torch.nn.functional as F



class PatchedSegResNetVAE(SegResNetVAE):
    def _get_vae_loss(self, net_input: torch.Tensor, vae_input: torch.Tensor):
        """
        Dynamically adapts to any input shape by flattening VAE features and adjusting FC layers.
        """
        x_vae = self.vae_down(vae_input)
        x_vae_flat = torch.flatten(x_vae, start_dim=1)

        # Dynamically re-initialize FC layers if needed
        if not hasattr(self, "_vae_fc_layers_initialized") or x_vae_flat.shape[1] != self.vae_fc1.in_features:
            in_features = x_vae_flat.shape[1]
            device = x_vae.device
            # print(f"[Patch] Adjusting VAE FC layers to in_features={in_features}")

            self.vae_fc1 = nn.Linear(in_features, self.vae_nz).to(device)
            self.vae_fc2 = nn.Linear(in_features, self.vae_nz).to(device)
            self.vae_fc3 = nn.Linear(self.vae_nz, in_features).to(device)

            self._vae_fc_layers_initialized = True

        z_mean = self.vae_fc1(x_vae_flat)
        z_mean_rand = torch.randn_like(z_mean).detach()  # random noise

        if self.vae_estimate_std:
            z_sigma = F.softplus(self.vae_fc2(x_vae_flat))
            vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)
            x_vae = z_mean + z_sigma * z_mean_rand
        else:
            vae_reg_loss = torch.mean(z_mean**2)
            x_vae = z_mean + self.vae_default_std * z_mean_rand

        x_vae = self.vae_fc3(x_vae)
        x_vae = self.act_mod(x_vae)

        # Unflatten using current x_vae_down shape
        vae_out_shape = list(self.vae_down(vae_input).shape)
        x_vae = x_vae.view(vae_out_shape)

        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = up(x_vae)
            x_vae = upl(x_vae)

        x_vae = self.vae_conv_final(x_vae)
        vae_mse_loss = F.mse_loss(net_input, x_vae)
        vae_loss = vae_reg_loss + vae_mse_loss
        return vae_loss

class nnUNetTrainerSegResNetVAE(nnUNetTrainerNoDeepSupervision):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.grad_scaler = None
        # self.initial_lr = 1e-4
        # self.weight_decay = 1e-5

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        
        label_manager = plans_manager.get_label_manager(dataset_json)

        model = PatchedSegResNetVAE(
            input_image_size=[
                128,
                160,
                112
            ], # M Plan; [128, 160, 128] #ResEnc XL
            
            spatial_dims = len(configuration_manager.patch_size),
            init_filters = 32,
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
        )

        return model
    

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # VAE loss
        output, vae_loss = self.network(data)
        l = self.loss(output, target)
        l += 0.1 * vae_loss 
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        output, _ = self.network(data)
        del data
        l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    # def configure_optimizers(self):

    #     optimizer = Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
    #     scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)

    #     return optimizer, scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass


class nnUNetTrainerSegResNet_100epochs(nnUNetTrainerSegResNetVAE):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
