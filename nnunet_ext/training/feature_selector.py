import itertools

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import umap
import logging
from typing import Tuple, Optional, Dict, List, Set
from collections import defaultdict
import torch.nn.functional as F
import warnings
from torch import nn

logger = logging.getLogger(__name__)


class FeatureSelector:
    def __init__(self,
                 random_state: int = 42,
                 batch_size: int = 1000,
                 min_cluster_size: int = 5,
                 min_samples: Optional[int] = None,
                 cluster_selection_epsilon: float = 0.0,
                 umap_n_neighbors: int = 15,
                 distance_type: str = 'manhattan',
                 min_dist: float = 0.1):
        """
        Initialize the feature selector with consistent distance metric usage.

        Args:
            distance_type: Distance metric to use throughout the analysis.
                         One of ['manhattan', 'correlation', 'cosine']
        """
        self.random_state = random_state
        self.batch_size = batch_size
        self.fitted_reducers = {}
        self.umap_n_neighbors = umap_n_neighbors
        self.distance_type = distance_type
        self.min_dist = min_dist

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon

        # Map manhattan to cityblock for scipy compatibility
        self._distance_mapping = {
            'manhattan': 'cityblock',
            'correlation': 'correlation',
            'cosine': 'cosine'
        }

    def _reduce_dimensions(self, features: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Dimension reduction using UMAP, with PCA fallback."""
        reducer_key = f'umap_{n_components}'

        if reducer_key not in self.fitted_reducers:
            n_samples = features.shape[0]

            if n_components >= n_samples:
                n_components = n_samples - 1

            try:
                # Try UMAP first
                self.fitted_reducers[reducer_key] = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=min(self.umap_n_neighbors, n_samples - 1),
                    min_dist=self.min_dist,
                    metric=self.distance_type,
                    random_state=self.random_state,
                    low_memory=False,
                    n_jobs=-1
                )
                return self.fitted_reducers[reducer_key].fit_transform(features)

            except (TypeError, ValueError) as e:
                # Fall back to PCA if UMAP fails
                print(f"UMAP failed: {e}. Falling back to PCA.")

                # Use PCA with same key so future calls use PCA consistently
                n_features = features.shape[1]
                max_components = min(n_samples, n_features)
                n_components = min(n_components, max_components)

                self.fitted_reducers[reducer_key] = PCA(
                    n_components=n_components,
                    random_state=self.random_state
                )
                return self.fitted_reducers[reducer_key].fit_transform(features)

        return self.fitted_reducers[reducer_key].transform(features)

    def select_samples(self,
                       features: torch.Tensor = None,
                       n_select: int = 30,
                       method: str = 'distance',
                       n_components_sampling: int = 100,
                       precomputed_embedding: Optional[np.ndarray] = None,
                       sample_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Select representative samples using specified method and configured distance metric.
        """

        if method == 'random':
            return self._select_random_samples(n_select, sample_ids)

        if features == None:
            assert "No features were provided to feature selector without choosing random sampling!"

        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        features_scaled = StandardScaler().fit_transform(features)

        raise ValueError(f"Unknown selection method: {method}")


    def fit_transform(self, features: torch.Tensor, n_components: int = 2) -> np.ndarray:
        """Main method to process and reduce dimensions."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Reduce dimensions with specified number of components
        return self._reduce_dimensions(features_scaled, n_components)


    def _select_random_samples(self, n_select: int,
                               sample_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Select random samples with equal distribution across patients.

        Args:
            n_select: Number of samples to select
            sample_ids: List of sample IDs in format 'DATASET_CaseX_Y'

        Returns:
            Tuple of (selected indices array, metrics dictionary)
        """
        n_samples = len(sample_ids)

        if n_samples <= n_select:
            # Fall back to selecting all samples
            rng = np.random.RandomState(self.random_state)
            selected_indices = np.sort(rng.choice(n_samples, size=min(n_select, n_samples), replace=False))
            return selected_indices, {
                'method': 'random',
                'selected_ratio': len(selected_indices) / n_samples
            }

        # Extract patient IDs from sample IDs
        patient_ids = []
        for sample_id in sample_ids:
            # Split by underscore and take the first part as patient ID
            patient_id = sample_id.split('_')[1]
            patient_ids.append(patient_id)

        # Get unique patients
        unique_patients = list(set(patient_ids))
        n_patients = len(unique_patients)

        # Calculate samples per patient
        base_samples_per_patient = n_select // n_patients
        extra_samples = n_select % n_patients

        rng = np.random.RandomState(self.random_state)
        selected_indices = []
        samples_per_patient = {}

        # Distribute extra samples randomly among patients
        if extra_samples > 0:
            extra_sample_patients = rng.choice(unique_patients, size=extra_samples, replace=False)
            samples_per_patient = {patient: base_samples_per_patient + (1 if patient in extra_sample_patients else 0)
                                   for patient in unique_patients}
        else:
            samples_per_patient = {patient: base_samples_per_patient for patient in unique_patients}

        # Select samples for each patient
        for patient in unique_patients:
            # Get indices for this patient
            patient_mask = [pid == patient for pid in patient_ids]
            patient_indices = np.where(patient_mask)[0]

            # Handle case where patient has fewer samples than allocated
            n_select_patient = min(samples_per_patient[patient], len(patient_indices))

            # Randomly select samples for this patient
            if len(patient_indices) > 0:
                selected_patient_indices = rng.choice(
                    patient_indices,
                    size=n_select_patient,
                    replace=False
                )
                selected_indices.extend(selected_patient_indices)

        # If we couldn't select enough samples due to patient constraints, throw assertion
        if len(selected_indices) < n_select:
            assert("We did not select as many features as we should for the patient!")

        metrics = {
            'method': 'stratified_random',
            'selected_ratio': len(selected_indices) / n_samples,
            'n_patients': n_patients,
            'target_samples_per_patient': samples_per_patient,
            'actual_samples_per_patient': {
                patient: sum(1 for idx in selected_indices if patient_ids[idx] == patient)
                for patient in unique_patients
            }
        }

        return np.sort(np.array(selected_indices)), metrics
