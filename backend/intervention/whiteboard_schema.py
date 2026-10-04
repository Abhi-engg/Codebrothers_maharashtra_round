from pydantic import BaseModel, field_validator, ValidationInfo
from typing import List, Literal, Optional, Union

class CoordinateMixin(BaseModel):
    @field_validator("x", "y", "x1", "y1", "x2", "y2", check_fields=False)
    @classmethod
    def check_normalized_coordinates(cls, v: float, info: ValidationInfo):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Coordinate {info.field_name} must be normalized between 0.0 and 1.0, got {v}")
        return v

class DrawLine(CoordinateMixin):
    tool: Literal["draw_line"] = "draw_line"
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = "#000000"
    width: int = 2
    dash: bool = False

class DrawArrow(CoordinateMixin):
    tool: Literal["draw_arrow"] = "draw_arrow"
    x1: float
    y1: float
    x2: float
    y2: float
    label: Optional[str] = None
    color: str = "#000000"

class DrawShape(CoordinateMixin):
    tool: Literal["draw_shape"] = "draw_shape"
    type: Literal["concave_mirror", "convex_lens", "resistor", "battery", "ray_beam"]
    x: float
    y: float
    width: float
    height: float
    style: Optional[str] = None

class DrawText(CoordinateMixin):
    tool: Literal["draw_text"] = "draw_text"
    x: float
    y: float
    text: str
    size: int = 14
    color: str = "#000000"
    math_flag: bool = False

class HighlightRegion(CoordinateMixin):
    tool: Literal["highlight_region"] = "highlight_region"
    x: float
    y: float
    width: float
    height: float
    label: Optional[str] = None

WhiteboardCommand = Union[DrawLine, DrawArrow, DrawShape, DrawText, HighlightRegion]

class WhiteboardSchema(BaseModel):
    commands: List[WhiteboardCommand]
