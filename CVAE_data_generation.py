## condition on product + EC, with regressor head
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances  # not used now; kept to minimize changes

# ----- Load Data -----
df = pd.read_csv('Datasets/example_data.csv')

mol2vec_cols = [f'mol2vec_{i}' for i in range(300)]
node2vec_cols = [f'Embedding_{i+1}' for i in range(128)]
ec2vec_cols = [f'ec2vec_{i}' for i in range(1024)]

x_cols = node2vec_cols
c_cols = mol2vec_cols + ec2vec_cols

X = df[x_cols].values.astype(np.float32)
C = df[c_cols].values.astype(np.float32)
y_o = df["kcat"].values.astype(np.float32)
y = np.log10(np.clip(y_o, 1e-6, None))

# ----- Define CVAEWithRegressor -----
class CVAEWithRegressor(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        self.kcat_head = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        y_pred = self.kcat_head(torch.cat([z, c], dim=1))
        return x_recon, mu, logvar, y_pred

# ----- Load Model -----
checkpoint = torch.load("Trained_model/cvae_model.pth",
                        map_location=torch.device('cpu'))
model = CVAEWithRegressor(input_dim=checkpoint['input_dim'],
                          cond_dim=checkpoint['condition_dim'],
                          latent_dim=checkpoint['latent_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ----- Setup -----
X_all = np.concatenate([C, X], axis=1)
nn_model_euclidean = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(X_all)
nn_model_cosine = NearestNeighbors(n_neighbors=5, metric='cosine').fit(X_all)

generated_embeddings, source_indices = [], []

z_values, mu_values, std_values, eps_values, alpha_values = [], [], [], [], []

# kNN labels (means), SEs (weighted), weighted std (sigma_kNN), and unweighted neighbor std
pseudo_kcats_soft_euclidean, standard_errors_soft_euclidean, sigma_knn_euclidean = [], [], []
neighbor_std_euclidean = []
pseudo_kcats_soft_cosine, standard_errors_soft_cosine, sigma_knn_cosine = [], [], []
neighbor_std_cosine = []

# Random baseline (unchanged)
pseudo_kcats_random, standard_errors_random = [], []

np.random.seed(42)

mol2vec_len = len(mol2vec_cols)
ec2vec_len  = len(ec2vec_cols)
node2vec_len = len(node2vec_cols)

n_replicates = 3
alpha_values_per_replicate = np.random.uniform(1.0, 5.0, size=n_replicates).tolist()

# ----- Helper: weighted kNN stats with effective sample size -----
def knn_stats(y_nn, distances, eps=1e-8):
    """
    Returns:
      mean_w (weighted mean with weights ~ 1/(d+eps)),
      se_w   (weighted SE using n_eff = 1/sum(w^2)),
      sigma_w (weighted std = sqrt(weighted variance)),
      std_unw (unweighted std of neighbor labels)
    """
    w = 1.0 / (distances + eps)
    w = w / w.sum()
    mean_w = np.sum(w * y_nn)
    wvar = np.sum(w * (y_nn - mean_w) ** 2)   # population weighted variance
    sigma_w = np.sqrt(wvar)                   # weighted std of neighbor labels
    neff = 1.0 / np.sum(w ** 2)               # effective sample size
    se_w = np.sqrt(wvar / max(neff, 1e-12))   # SE of weighted mean
    std_unw = np.std(y_nn)                    # unweighted std of the 5 neighbors
    return mean_w, se_w, sigma_w, std_unw

# ----- Generate + Label -----
model.eval()
with torch.no_grad():
    for i in range(len(X)):
    
        if i % 500 == 0:
           print(i)

        x_i = torch.tensor(X[i]).unsqueeze(0)
        c_i = torch.tensor(C[i]).unsqueeze(0)
        mu, logvar = model.encode(x_i, c_i)

        for rep_id in range(n_replicates):
            alpha = alpha_values_per_replicate[rep_id]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + alpha * eps * std

            z_values.append(z.squeeze(0).cpu().numpy())
            mu_values.append(mu.squeeze(0).cpu().numpy())
            std_values.append(std.squeeze(0).cpu().numpy())
            eps_values.append(eps.squeeze(0).cpu().numpy())
            alpha_values.append(alpha)

            x_gen = model.decode(z, c_i).squeeze(0).numpy()

            node2vec_part = x_gen[:node2vec_len]
            mol2vec_part  = c_i[0, :mol2vec_len].numpy()
            ec2vec_part   = c_i[0, mol2vec_len:].numpy()
            full_instance = np.concatenate([mol2vec_part, ec2vec_part, node2vec_part])

            generated_embeddings.append(full_instance)
            source_indices.append(i)

            # tiny epsilon for distance weights (avoid division by zero)
            eps_small = 1e-8
            k = 5

            # ----- Soft kNN using Euclidean -----
            distances_eu, nn_indices_eu = nn_model_euclidean.kneighbors(
                full_instance.reshape(1, -1), return_distance=True
            )
            distances_eu = distances_eu[0]; nn_indices_eu = nn_indices_eu[0]
            y_nn_eu = y_o[nn_indices_eu]

            kcat_eu, se_eu, sigma_eu, std_eu = knn_stats(y_nn_eu, distances_eu, eps=eps_small)
            pseudo_kcats_soft_euclidean.append(kcat_eu)
            standard_errors_soft_euclidean.append(se_eu)
            sigma_knn_euclidean.append(sigma_eu)
            neighbor_std_euclidean.append(std_eu)

            # ----- Soft kNN using built-in Cosine -----
            distances_cos, nn_indices_cos = nn_model_cosine.kneighbors(
                full_instance.reshape(1, -1), return_distance=True
            )
            distances_cos = distances_cos[0]; nn_indices_cos = nn_indices_cos[0]
            y_nn_cos = y_o[nn_indices_cos]

            kcat_cos, se_cos, sigma_cos, std_cos = knn_stats(y_nn_cos, distances_cos, eps=eps_small)
            pseudo_kcats_soft_cosine.append(kcat_cos)
            standard_errors_soft_cosine.append(se_cos)
            sigma_knn_cosine.append(sigma_cos)
            neighbor_std_cosine.append(std_cos)

            # ----- Random baseline (unchanged) -----
            rand_indices = np.random.choice(len(X), size=k, replace=False)
            y_rand = y_o[rand_indices]
            pseudo_kcats_random.append(y_rand.mean())
            standard_errors_random.append(y_rand.std() / np.sqrt(k))

# ----- Save Results -----
df_generated = pd.DataFrame(generated_embeddings, columns=mol2vec_cols + ec2vec_cols + node2vec_cols)

df_generated["kcat_euclidean"] = pseudo_kcats_soft_euclidean
df_generated["se_euclidean"] = standard_errors_soft_euclidean       # weighted SE
df_generated["sigma_knn_euclidean"] = sigma_knn_euclidean                  # weighted std
df_generated["std_neighbors_euclidean"] = neighbor_std_euclidean               # unweighted std

df_generated["kcat_cosine"] = pseudo_kcats_soft_cosine
df_generated["se_cosine"] = standard_errors_soft_cosine          # weighted SE
df_generated["sigma_knn_cosine"] = sigma_knn_cosine                     # weighted std
df_generated["std_neighbors_cosine"] = neighbor_std_cosine                  # unweighted std

df_generated["kcat_random"] = pseudo_kcats_random
df_generated["se_random"] = standard_errors_random

df_generated["source_index"] = source_indices
df_generated["z"] = z_values
df_generated["mu"] = mu_values
df_generated["std"] = std_values
df_generated["eps"] = eps_values
df_generated["alpha"] = alpha_values

df_generated.to_csv(f"Synthetic_data/generated_synthetic_data_{n_replicates}_per_original_instance.csv", index=False)
