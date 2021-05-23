from typing import List, Union

from pydantic import BaseModel, conlist, validator

from .feature_params import INPUT_FEATURES_LIST


class AppRequest(BaseModel):
    data: List[conlist(Union[float, str], min_items=1)]  # max_items=50
    features: List[str]

    @validator("features")
    def validate_model_features(cls, features):
        if features != INPUT_FEATURES_LIST:
            raise ValueError(f"Incorrect feature count or order!\n"
                             f"Expected: {INPUT_FEATURES_LIST}")
        return features


class AppResponse(BaseModel):
    id: str
    disease: int
