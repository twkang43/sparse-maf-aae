import time
import pandas
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from SparseMAFAAE.MAF import MAF
from SparseMAFAAE.Generator import Generator
from SparseMAFAAE.Discriminator import SimpleDiscriminator

from utils import set_device, sliding_window, get_metrics
from data_preprocessing.voraus_ad import ANOMALY_CATEGORIES

DEVICE = set_device.set_device()
EVAL_TIME = 20

# Train & Eval Adversarial Autoencoders
class AAE():
    def __init__(self, args, config, dataset):
        self.train_loader, self.test_loader = dataset
        self.initialize_variables(args, config)

        data = next(iter(self.train_loader))
        self.io_size = data[0].shape[2]
        self.latent_size = self.io_size*2

        self.set_models()
        self.set_optimizers()

        # Print the number of the model's parameters
        print(f"# of parameters - Generator : {sum(p.numel() for p in self.generator.parameters() if p.requires_grad)}")
        print(f"# of parameters - Discriminator : {sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)}")

    def initialize_variables(self, args, config):
        self.features, self.frequency, self.subset_size = args.features, args.frequency, args.subset_size

        self.lr, self.epochs, self.batch_size = config.lr, config.epochs, config.batch_size
        self.window_size, self.win_stride = config.window_size, config.win_stride

        self.gamma = config.gamma
        self.beta, self.max_beta, self.l1_weight = 0.0, config.max_beta, config.l1_weight

    def set_models(self):
        self.generator = Generator(
            io_size     = self.io_size,      # N
            hidden_size = self.io_size*2,    # H
            latent_size = self.io_size*2,    # L
            num_layers  = 2,
            num_flows   = 4,
            data_len    = self.window_size,  # D
        ).to(DEVICE)

        self.prior_model = MAF(data_len=self.window_size, num_flows=4).to(DEVICE)
        self.prior_model.load_state_dict(self.generator.encoder.maf.state_dict())

        self.discriminator = SimpleDiscriminator(self.latent_size).to(DEVICE)
        self.discriminator.train()

        self.true_label = torch.ones((self.window_size,1), requires_grad=False, device=DEVICE)
        self.fake_label = torch.zeros((self.window_size,1), requires_grad=False, device=DEVICE)

    def set_optimizers(self):
        # Generator
        self.G_optimizer = optim.AdamW(params=self.generator.parameters(), lr=self.lr)
        self.G_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.G_optimizer, milestones=[self.epochs//6, 5*self.epochs//6], gamma=self.gamma)

        # Discriminator
        self.D_optimizer = optim.AdamW(params=self.discriminator.parameters(), lr=self.lr)
        self.D_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.D_optimizer, milestones=[self.epochs//6, 5*self.epochs//6], gamma=self.gamma)

    def exp_decay_annealing(self, epoch, max_beta, k_factor=2.0):
        k = torch.tensor(k_factor/self.epochs, device=DEVICE)
        return max_beta * (1 - torch.exp(-k*epoch))

    def get_generator_loss(self, x_hat, x, mean, log_var):
        # Reconstruction Loss
        mse = F.mse_loss(x_hat, x, reduction="sum")
        
        # Generator Loss
        z0 = self.generator.encoder.reparameterization(mean, log_var)
        G_posterior = self.generator.encoder.maf(z0)
        generator_loss = F.binary_cross_entropy(self.discriminator(G_posterior), self.true_label)

        # L1 Loss
        l1_loss = torch.tensor(0.0, requires_grad=True, device=DEVICE)
        if 0.0 < self.l1_weight:
            for param in self.generator.encoder.parameters():
                # Add l1 norms to l1_loss
                l1_loss = l1_loss + torch.norm(param, p=1)
            l1_loss = l1_loss * self.l1_weight
        
        return mse + self.beta*generator_loss + l1_loss
    
    def get_discriminator_loss(self, mean, log_var):
        # Samples
        prior = self.prior_model(torch.randn((self.window_size, self.latent_size), device=DEVICE))
        D_posterior = self.generator.encoder.maf(self.generator.encoder.reparameterization(mean, log_var))

        # Discriminator Loss
        true_loss = F.binary_cross_entropy(self.discriminator(prior), self.true_label)
        fake_loss = F.binary_cross_entropy(self.discriminator(D_posterior.detach()), self.fake_label)
        return 0.5 * (true_loss + fake_loss)
    
    # Train AAE models
    def train(self):
        self.generator.train()
        batch_count = 0
        dir = f"./logs/voraus-AD-{self.features}-{self.frequency}hz-{self.subset_size}/training"
        writer = SummaryWriter(log_dir=dir)

        for epoch in tqdm(range(self.epochs), desc="Total "):
            self.beta = self.exp_decay_annealing(epoch, self.max_beta)

            # Update a prior model state
            self.prior_model.load_state_dict(self.generator.encoder.maf.state_dict())

            for batch_num, (batch, _) in enumerate(tqdm(self.train_loader, desc="Epochs", leave=False)):
                batch = batch.to(DEVICE) # B x T x N
                batch_count += batch_num
                update_GEN, update_DIS = [], []

                for _, x in enumerate(batch):
                    x_window = sliding_window.generate_sliding_window(x, self.window_size, self.win_stride)

                   # Process each window for the Discriminator
                    for x_partial in x_window:
                        x_hat, _, (mean, log_var) = self.generator(x_partial)

                        generator_loss = self.get_generator_loss(x_hat, x_partial, mean, log_var)
                        update_GEN.append(generator_loss)

                        discriminator_loss = self.get_discriminator_loss(mean, log_var)
                        update_DIS.append(discriminator_loss)

                # Update Generator weights after processing the whole batch
                self.G_optimizer.zero_grad()
                mean_gen = torch.mean(torch.stack(update_GEN))
                mean_gen.backward()
                self.G_optimizer.step()

                # Update Discriminator weights after processing the whole batch
                self.D_optimizer.zero_grad()
                mean_dis = torch.mean(torch.stack(update_DIS))
                mean_dis.backward()
                self.D_optimizer.step()

                # Add losses to TensorBoard
                writer.add_scalar("Generator Loss", mean_gen.detach().clone().cpu(), batch_count)
                writer.add_scalar("Discriminator Loss", mean_dis.detach().clone().cpu(), batch_count)

                # Reset lists for the next batch
                update_GEN.clear()
                update_DIS.clear()

            # Update learning rates
            self.G_scheduler.step()
            self.D_scheduler.step()

            # Validate
            auroc = self.validate()
            writer.add_scalar("AUROC", auroc, epoch)

        writer.close()

    def validate(self):
        self.generator.eval()
        l1_mean, l1_std = self.get_l1_statistics()

        with torch.no_grad():
            result_list = []
            for _, (x, labels) in enumerate(tqdm(self.test_loader, desc="Validating...", leave=False)):
                x = x.to(DEVICE) # 1 x T x N
                x = x.squeeze(axis=0) # T x N
                partial_list = []

                # Sliding Window
                x_window = sliding_window.generate_sliding_window(x, self.window_size, self.win_stride)
    
                for _, x_partial in enumerate(x_window):
                    # Calculate l1 norms
                    x_hat, _, _ = self.generator(x_partial)
                    l1_norm = torch.norm(x_partial-x_hat, p=1)

                    # Calculate the anomaly score per sample.
                    z_score = (l1_norm - l1_mean) / l1_std
                    z_score = z_score.unsqueeze(dim=0).to(DEVICE)
                    partial_list.append(z_score)
                        
                # Append the anomaly score and the labels to the results list.
                result_labels = {k: v[0].item() if isinstance(v, torch.Tensor) else v[0] for k, v in labels.items()}
                result_labels.update(score=max(partial_list).item())
                result_list.append(result_labels)

        results = pandas.DataFrame(result_list)

        aurocs = []
        for category in ANOMALY_CATEGORIES:
            dfn = results[(results["category"] == category.name) | (~results["anomaly"])]

            fpr, tpr, _ = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
            auroc = metrics.auc(fpr, tpr)*100
            aurocs.append(auroc)

        self.generator.train()
        return np.mean(aurocs)
    
    # Get mean & std of valid dataset
    def get_l1_statistics(self):
        with torch.no_grad():
            l1_norms = []

            for _, (x, label) in enumerate(tqdm(self.test_loader, desc="Calculating z_score...", leave=False)):
                if label["anomaly"].item():
                    continue

                x = x.to(DEVICE) # 1 x T x N
                x = x.squeeze(axis=0) # T x N

                x_window = sliding_window.generate_sliding_window(x, self.window_size, self.win_stride)
                
                for _, x_partial in enumerate(x_window):
                    # Calculate l1 norms
                    x_hat, _, _ = self.generator(x_partial)
                    l1_norms.append(torch.norm(x_partial-x_hat, p=1))

            # Calculate z-score
            l1_tensors = torch.tensor(l1_norms, device=DEVICE)
            l1_mean = torch.mean(l1_tensors)
            l1_std = torch.std(l1_tensors)

        return l1_mean, l1_std

    # Evaluate AE models
    def eval(self):
        self.generator.eval()
        aurocs_list, elapsed_time_list = [], []

        for i in range(EVAL_TIME):
            l1_mean, l1_std = self.get_l1_statistics()
            
            with torch.no_grad():
                result_list, elapsed_time = [], []
                # Get a data to generate a ROC & Precision-Recall Curve
                for _, (x, labels) in enumerate(tqdm(self.test_loader, desc="Evaluating...", leave=False)):
                    x = x.to(DEVICE) # 1 x T x N
                    x = x.squeeze(axis=0) # T x N
                    partial_list = []

                    # Sliding Window
                    x_window = sliding_window.generate_sliding_window(x, self.window_size, self.win_stride)
    
                    for _, x_partial in enumerate(x_window):
                        # Calculate l1 norms
                        start_time = time.time()
                        x_hat, _, _ = self.generator(x_partial)
                        l1_norm = torch.norm(x_partial-x_hat, p=1)

                        # Calculate the anomaly score per sample.
                        z_score = (l1_norm - l1_mean) / l1_std
                        z_score = z_score.unsqueeze(dim=0)

                        # Calculate the elapsed time
                        end_time = time.time()
                        elapsed_time.append((end_time-start_time)*1000)

                        partial_list.append(z_score)
                        
                    # Append the anomaly score and the labels to the results list.
                    result_labels = {k: v[0].item() if isinstance(v, torch.Tensor) else v[0] for k, v in labels.items()}
                    result_labels.update(score=max(partial_list).item())
                    result_list.append(result_labels)

            print(f"#{i:<2} - ", end="")
            results = pandas.DataFrame(result_list)
            aurocs = get_metrics.get_metrics(results)
            time_within_IQR = get_metrics.calculate_elapsed_time(elapsed_time)

            aurocs_list.append(aurocs)
            elapsed_time_list.append(time_within_IQR)

        aurocs_np = np.array(aurocs_list)
        elapsed_time_np = np.array([item for sublist in elapsed_time_list for item in sublist])

        print()
        for i, category in enumerate(ANOMALY_CATEGORIES):
            print(f"{category.name:<17} : {np.mean(aurocs_np[:,i]):.2f} \u00B1 {np.std(aurocs_np[:,i]):.2f}")

        print(f"\nOverall performance : {np.mean(aurocs_np):.2f} \u00B1 {np.std(aurocs_np, ddof=1):.2f}")
        print(f"Overall elapsed time : {np.mean(elapsed_time_np):.4f} \u00B1 {np.std(elapsed_time_np):.4f}")