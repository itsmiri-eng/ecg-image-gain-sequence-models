import time
from pathlib import Path

import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


class LDAPredictor:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, images):
        return self.pipeline.predict_proba(images.reshape(len(images), -1))


def train_lda(x_train, y_train, config, seed, output_dir):
    if config["priors"] != "uniform":
        raise ValueError("This prespecified baseline supports uniform priors only")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "pipeline.joblib"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    n_components = min(config["pca_components"], len(x_train) - 4, x_train[0].size)
    pipeline = Pipeline([
        ("pca", PCA(n_components=n_components, svd_solver=config["pca_solver"], random_state=seed)),
        ("lda", LinearDiscriminantAnalysis(solver=config["solver"], shrinkage=config["shrinkage"],
                                           priors=[0.25] * 4)),
    ])
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        pipeline.fit(x_train.reshape(len(x_train), -1), y_train)
    seconds = time.perf_counter() - start
    joblib.dump(pipeline, checkpoint)
    return LDAPredictor(pipeline), {
        "fit_seconds": seconds, "device": "cpu", "seed": seed,
        "input_features": int(x_train[0].size), "pca_components": n_components,
        "pca_explained_variance_ratio_sum": float(pipeline["pca"].explained_variance_ratio_.sum()),
        "preprocessing": "Fixed cropped grayscale ink [0,1]; flatten; train-only centered PCA; shrinkage LDA",
        "feature_scaling": "No per-pixel variance scaling: all pixel intensities share the same units",
        "class_imbalance": "Uniform class priors, prespecified to emphasize all four classes equally",
        "checkpoint": str(checkpoint), "config": config,
    }
