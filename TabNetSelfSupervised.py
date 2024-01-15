import torch
import numpy as np
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.preprocessing import LabelEncoder

class TabNetSelfSupervised:
    def __init__(self, n_d=16, n_a=16, n_steps=4, n_independent=2, n_shared=2, cat_emb_dim=2, 
                 optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2), mask_type="entmax"):
        """
        Parameters:
        - n_d, n_a, n_steps, n_independent, n_shared: TabNet architecture parameters
        - cat_emb_dim: Dimensionality of the embeddings for categorical features
        - optimizer_fn: Optimizer function to use for training
        - optimizer_params: Dictionary containing optimizer parameters
        - mask_type: Type of mask to use during training ("sparsemax" or "entmax")
        """
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.cat_emb_dim = cat_emb_dim
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.mask_type = mask_type
        self.pretrainer = None
        self.label_encoders = {}

    def fit(self, X, cat_features):

        cat_dims = [len(np.unique(X[:, col_idx])) for col_idx in cat_features]

        self.pretrainer = TabNetPretrainer(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            cat_idxs=cat_features,
            cat_dims=cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            mask_type=self.mask_type
        )

        #X = X[:1000]
        
        self.pretrainer.fit(
            X_train=X,
            eval_set=[X],
            max_epochs=10,  # You can adjust this
            batch_size=256,  # You can adjust this
            virtual_batch_size=128,  # You can adjust this
            num_workers=0  # You can adjust this
        )

    def transform(self, X):

        if self.pretrainer is None:
            raise ValueError("The pretrainer has not been fitted. Please fit the pretrainer first.")

        # Label encode categorical features using the saved encoders
        for col_idx, le in self.label_encoders.items():
            X[:, col_idx] = le.transform(X[:, col_idx])

        # Use the pretrainer to get representations
        _, representations = self.pretrainer.predict(X)
                
        return representations