from dataclasses import dataclass
from typing import List, Optional


INPUT_FEATURES_LIST = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                       "thalach", "exang", "oldpeak", "slope", "ca", "thal", "id"]


@dataclass()
class FeatureParams:
    numerical_features: List[str]
    categorical_features: List[str]
    target_col: Optional[str]
    age_threshold: Optional[int]
