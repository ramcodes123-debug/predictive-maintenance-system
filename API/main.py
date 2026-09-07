from fastapi import FastAPI
from pydantic import BaseModel, Field
import sys
from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import prediction function
from SRC.predict import predict


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "REST API for predicting industrial machine failure "
        "using the AI4I 2020 Predictive Maintenance dataset."
    ),
    version="1.0.0"
)


# =========================================================
# INPUT SCHEMA
# =========================================================

class MachineInput(BaseModel):

    air_temp: float = Field(
        ...,
        description="Air temperature in Kelvin"
    )

    process_temp: float = Field(
        ...,
        description="Process temperature in Kelvin"
    )

    rot_speed: float = Field(
        ...,
        description="Rotational speed in rpm"
    )

    torque: float = Field(
        ...,
        description="Torque in Nm"
    )

    tool_wear: float = Field(
        ...,
        description="Tool wear in minutes"
    )

    type: str = Field(
        "M",
        description="Machine type: L, M, or H"
    )


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Predictive Maintenance API is running",
        "status": "success"
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def make_prediction(machine: MachineInput):

    raw_input = {
        "air_temp": machine.air_temp,
        "process_temp": machine.process_temp,
        "rot_speed": machine.rot_speed,
        "torque": machine.torque,
        "tool_wear": machine.tool_wear,
        "type": machine.type.upper()
    }

    result = predict(raw_input)

    return result