import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
# from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       
        
        ######################################################################
        # # Generate parameters for t_s as a "scalar"
        # betaList = beta_1 + (beta_T - beta_1) * torch.linspace(0, 1, T)  # Shape (T,)
        # beta_t_range = betaList[:t_s]
        # alpha_t_range = 1 - beta_t_range
        
        # beta_t = beta_t_range[t_s - 1]
        # sqrt_beta_t = torch.sqrt(beta_t)
        # alpha_t = 1 - beta_t
        # oneover_sqrt_alpha = 1/torch.sqrt(alpha_t)
        
        # alpha_t_bar = torch.prod(alpha_t_range)
        # sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        # sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)
        ######################################################################
        
        if t_s.dim() == 0:     # Scalar: reshape to (1, 1)
            t_s = t_s.view(1, 1)
        B = t_s.size(0)
        
        sch_device = t_s.device # define the scheduling parameters on the device
        # Should be cuda for all cases, but unit testing cases created on cpu.
        
        # Get the range of variance schedule and calculate beta_t for all batch samples
        betas = beta_1 + (beta_T - beta_1) * torch.linspace(0, 1, T, device = sch_device)  # Shape (T,)
        beta_expanded = betas.expand(B, -1)  # (B, T)
        beta_t = torch.gather(beta_expanded, 1, t_s-1) # (B, 1)
        sqrt_beta_t = torch.sqrt(beta_t)
        
        alpha_t = 1 - beta_t  # (B, 1)
        oneover_sqrt_alpha = 1 / torch.sqrt(alpha_t)  # Shape (B, 1)
        
        # Build a mask: for each row (sample in batch), mark alphas up to t_s-th element
        alpha_expanded = 1 - beta_expanded # (B, T)
        row_indices = torch.arange(T, device = sch_device).unsqueeze(0).expand(B, -1)  # (B, T)
        mask = row_indices <= t_s-1  # mask of shape (B, T)
        alpha_cum = torch.where(mask, alpha_expanded, torch.ones_like(beta_expanded))
        alpha_t_bar = alpha_cum.prod(dim=1, keepdim=True)  # (B, 1)

        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)  # Shape (B, 1)
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)  # Shape (B, 1)
        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  
        B = images.size(0)
        H, W = images.size(2), images.size(3)
        num_drop = int(self.dmconfig.mask_p*B)
        drop_idx = torch.randperm(B)[:num_drop]
                       
        con = F.one_hot(conditions, self.dmconfig.num_classes)
        con[drop_idx] = -1
        t = torch.randint(1, T + 1, (B, 1), device = device)
        # Dictionary of schedule parameters: each parameter of shape (B, 1) for each sample in batch
        schedule_param = self.scheduler(t) 
        noise = torch.randn(B, 1, H, W, device = device)
        
        a = schedule_param["sqrt_alpha_bar"]
        one_minus_a = schedule_param["sqrt_oneminus_alpha_bar"]
        images_t = a.view(B, 1, 1, 1)*images + one_minus_a.view(B, 1, 1, 1)*noise
        noise_diff = (self.network(images_t, t.view(B, 1, 1, 1)/T, con) - noise) 
        noise_loss = (noise_diff** 2).sum()/B
        
        # self.network (inputs):
        #     x: input images, with size (B,1,28,28)
        #     t: input time stepss, with size (B,1,1,1)
        #     c: input conditions (one-hot encoded labels), with size (B,10)
        # ==================================================== #
        
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  
        
        B = conditions.size(0)
        con = F.one_hot(conditions, self.dmconfig.num_classes)
        uncon = torch.full((B, 10), -1, device = device)
        
        X_t = torch.randn(B, 1, 28, 28, device = device)
        
        with torch.no_grad():
            for t in reversed(range(1, T+1)):
                noise = torch.randn_like(X_t)
                if t == 0:
                    noise.zero_()
                
                t_input = torch.full((B, 1), t, device = device)
                schedule_param = self.scheduler(t_input)
        
                oneover_sqrt = schedule_param["oneover_sqrt_alpha"].view(B, 1, 1, 1)
                beta = schedule_param["beta_t"].view(B, 1, 1, 1)
                sqrt_oneminus_alpha_bar = schedule_param["sqrt_oneminus_alpha_bar"].view(B, 1, 1, 1)
                sqrt_beta = schedule_param["sqrt_beta_t"].view(B, 1, 1, 1)
                
                epsilon = (1 + omega)*self.network(X_t, t_input/T, con) - omega*self.network(X_t, t_input/T, uncon)
                X_t = oneover_sqrt*(X_t - beta / sqrt_oneminus_alpha_bar * epsilon) + sqrt_beta*noise
        
        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images