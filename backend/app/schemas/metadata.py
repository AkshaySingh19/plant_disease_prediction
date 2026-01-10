from typing import Dict, List
from pydantic import RootModel, field_validator


class MetadataSchema(RootModel[Dict[str, List[str]]]):
    """
    Schema to validate metadata.json structure.
    """

    @field_validator("root")
    @classmethod
    def validate_metadata(cls, value: Dict[str, List[str]]):
        if not value:
            raise ValueError("metadata.json can't be empty")

        for crop, classes in value.items():

            # crop name validation
            if not isinstance(crop, str) or not crop.strip():
                raise ValueError(f"Invalid crop name: {crop}")

            # class list validation
            if not isinstance(classes, list) or len(classes) == 0:
                raise ValueError(
                    f"Class list for crop '{crop}' must be a non-empty list"
                )

            # each class must be string
            for label in classes:
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(
                        f"Invalid class label '{label}' in crop '{crop}'"
                    )

        return value
