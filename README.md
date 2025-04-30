# What is Two MLOps Toolset?

This repo contains a toolset for streamlined training and serving of ML models.

## Structure of the repo

This repo consist of the following sections:

1. `src`: The source code which itself has two parts: `train` and `serve`. Ideally, these two sections should be one repo each, but for the simplicity ke keep them together here.
2. `tests`: The automated tests for quality check of the code which has two sections: `unit_tests` and `integration_tests`.
3. `artifacts`: which is used for storing the ML model artifacts and their metadata after the pipeline is run. Ideally, one should use some artifact management service to store the artifacts in order to avoid commiting big files to git history, but we keep it here for simplicity.
4. `Dockerfile` and `docker-compose.yaml` for containerizing the API after we are happy with it and streamlining the local build and run processes.
5. `.github`: for creating three code quality check pipelines: `code-testing` (running unit tests), `code-validation`: (running ruff for linting and formatting), and `code-security` (running bandit on the source code and pip-audit on the dependencies).
6. `requirements.txt` and `validation_requirements.txt` which are the dependencies required for running the source code and the pipelines, respectively.
7. The rest of the structure is self-explanatory and standard, e.g. `.pre-commit-config.yaml`, for adding pre-commit hooks, `noxfile.py` for isolated python sessions (for linting, testing, etc.), `pyprojct.toml` for setting up the tools used and packaging the code if needed, etc.

## How to run the project

This section gives a step by step guideline on how to run the task.

### Installation

Even though the source code (`src`) is written as a package, you do not need to install it. All you need for running the training pipeline, deploying the service, running unit and integration tests, etc. is to install dependencies after cloning the repo (and creating your virtual environment).

In order to do so:

```bash
pip install -r requirements.txt
pip install -r validation_requirements.txt
```

## Training pipeline

The following architectural decisions are made here which are followed by the rationale behind them.

1. The training logic is defined as a series of functions which are all concatenated under one class (e.g. `IrisClassifier`). Doing so makes the code more modular, more understandable from holistic point of view, and also better unit-testable.

2. Io order to be standard, one abstract class is defined from which all the training classes should enherit. This helps to develop also a standard and reusable code for training pipeline.

3. The training pipeline is designed using metaflow, and it is added as part of the package. So basically, for every new training one needs a config file (e.g. `src/training/config.yaml`), and the following lines of code:

```bash
from src.training.training_workflow import TrainingWorkflow

TrainingWorkflow()
```

### Running the training pipeline

In order to run the training pipeline and obtain the serialized ML model and preprocessing pipeline, you should run:

```bash
python run_training_workflow.py run --config src/training/config.yaml
```

By running this command, the workflow should be run and you should get the serialized model and its metadata stored in `artifacts/`.

NOTE: I just noticed the config file as is, does not load the data from input (since it is using the iris data). However, one can simply enherit a class from IrisClassifier modify only the `load_data` method (3 lines of code), register the new class to the workflow_classes using the `register_workflow` decorator and add the name of the new class and the data path to the `config.yaml` file. And you are good to go.

### Testing the training pipeline

Defining the training steps as methods of a class, we can unit test them. A series of unit tests are developed in `tests/unit_tests/training/`, and one can run them by:

```bash
pytest tests/unit_tests/training/
```

## Serving the model

In order to serve the ML model, a FastAPI app is designed to serve the ML model and is run using Uvicorn. Similar to the training pipeline, it is tried to be modular. A dummy authentication mechanism is included which can substituted by JWT in live.

In order to run the API locally, you can set the corresponding config file (e.g. `src/serve/config.yaml`) set the `FASTAPI_PORT` and run the app:

```bash
export FASTAPI_PORT=5050
python -m src.serve.app
```

Then you can simply test the API:

```bash
# For health check
curl -i http://localhost:5050/health

# For predict
curl -i -X POST http://localhost:5050/predict   -H "Content-Type: application/json"   -d '{
       "request_id": "integration-test-id",
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
      }'
```

Or with better frameworks like `Postman` if you wish.

### Testing the API
In order to test the API some example unit tests are added to `tests/unit_tests/serve/`, which can be run:

```bash
pytest tests/unit_tests/serve/
```

## CI pipelines

The three CI pipelines are automatically run when a Pull Request is created for the `main` branch, and do the following tasks:

1. `code-testing`: runs all the unit
2. `code-validation`: Runs ruff for linting and formatting, (just a template more can be done here)
3. `code-security`: Runs bandit on the source code and pip-audit on the dependencies (also a template more can be done)

In addition some automation shell scripts are added for streamlining the quality check procedure (e.g. `.github/scripts/check_format.sh`, `.github/scripts/security.sh`). They are used together with `noxfile.py` in the pipelines.

## Containerization

In order for containerization of the app, we use docker. A simple `Dockerfile` is added which can be used by the settings in `docker-compose.yaml` for streamlined deployment on the local machine.

```bash
# To build
docker compose build ml-api-service

# To run
docker compose up ml-api-service
```

## Integration test

Some integration tests are developed for testing the container on the local machine, after building the docker image, you can run them using:
```bash
pytest tests/integration_tests/
```

## Versioning

We use double-versioning in this repo: one for the ML model, and one for the API. The ML model version should be stored in the metadata coming with the model, and the API version is git-tagged on the corresponding commit in the repo. The image version (pushed to some container registry) will be the same as the API version.

The response payload will include both the ML model version and the API version (this is another small change compared to the assignment itself). This helps with better monitoring in the downstream; e.g. only comparing different ML models, not APIs.

## CD pipeline

From this point on, the project is explained conceptually. The CD part is done in the following steps using IaC and GitOps.

1. Creating the required infrastrucure (container registry, kubernetes cluster, dev, perf-test, live environments, ...) using IaC (e.g. Terraform)
2. Creating CI pipelines for building and pushing the API image to the container registry
3. Creating test workflows for all pre-prod environments (dev, perf-test, acceptance, ...) and making sure the corresponding CD tests (integration test, load test, ...) for each environment are passed, before deployment to live.
4. Implementing CD pipelines with GitOps for managing deployments across all environments, e.g. using ArgoCD.

## Logging and Monitoring

Both technical monitoring and business monitoring can be explained briefly here.

As for technical monitoring there are several choices depending on the need, scale, budget, ... (e.g. Datadog, NewRelic, Prometheus, ...). These tools can track metrics such as CPU usage, memory consumption, request latency, error rates, etc. In the code provided, a basic prototype of using Prometheus is provided.

As for business monitoring, one can use tools like Grafana to create dashboards for business monitoring (transaction volume, revenue generation, ...) in realtime.

As for logging, loguru is used because of its ease of use, but proper logging and log-handling should be added on live.
