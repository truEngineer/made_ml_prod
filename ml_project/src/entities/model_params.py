from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    model_type: str = field(default="LogisticRegression")
    inv_regularization_strength: float = field(default=1.0)
    intercept_scaling: float = field(default=1.0)
    n_estimators: int = field(default=200)
    random_state: int = field(default=42)
