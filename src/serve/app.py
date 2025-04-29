"""A FastAPI application for serving a machine learning model."""

import joblib
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from src.serve.api_utils.authentication import dummy_authenticator
from src.serve.api_utils.base_app import router as health_router
from src.serve.api_utils.config import load_config
from src.serve.api_utils.schemas import IrisRequest, IrisResponse

# Load configuration
config = load_config()

logger.info(f"Loading model from {config.model.path}")
model = joblib.load(config.model.path)


app = FastAPI(
    title="Iris Inference Service",
    version=config.version,
    description="Predict Iris species based on flower measurements.",
)

Instrumentator().instrument(app).expose(app)

# Include the health check router
app.include_router(health_router)


@app.post("/predict", response_model=IrisResponse, summary="Predict Iris Species")
async def predict(
    request: Request,
    data: IrisRequest,
    verified_token: bool = Depends(dummy_authenticator),
):
    """
    Predict the species of Iris flower based on its measurements.

    Args:
    ----
        request (Request): The HTTP request object.
        data (IrisRequest): The request body containing flower measurements.
        verified_token (bool): The result of the authentication check.

    Returns:
    -------
    IrisResponse: The prediction result including species and model version.

    """
    request_id = data.request_id

    if not verified_token:
        logger.warning(f"Unauthorized access attempt. Request ID: {request_id}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    try:
        features = np.array(
            [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
        )
        pred_idx = int(model.predict(features)[0])

        response = IrisResponse(
            prediction=pred_idx,
            prediction_label=config.model.species[pred_idx],
            request_id=request_id,
            model_version=config.model.version,
            api_version=config.version,
        )

        logger.info(
            "Prediction successful. Request ID: "
            f"{request_id} Prediction: {response.prediction_label}"
        )
        return response

    except Exception as e:
        logger.exception(f"Prediction failed. Request ID: {request_id}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}"
        ) from e


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
    )
