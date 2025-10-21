import requests
from datetime import datetime
import uuid


def get_run_via_rest(run_id, tracking_uri="http://127.0.0.1:5000"):
    """Fetch run info dynamically from the MLflow Tracking Server REST API."""
    url = f"{tracking_uri}/api/2.0/mlflow/runs/get?run_id={run_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["run"]


def dynamic_mlflow_to_tmf(run_id, tracking_uri="http://127.0.0.1:5000"):
    """Convert MLflow run → TMF-compliant JSON dynamically via MLflow REST API."""
    run = get_run_via_rest(run_id, tracking_uri)
    info = run["info"]
    data = run["data"]

    def list_to_dict(items):
        if isinstance(items, list):
            return {item["key"]: item["value"] for item in items}
        return items

    params = list_to_dict(data.get("params", {}))
    metrics = list_to_dict(data.get("metrics", {}))
    tags = list_to_dict(data.get("tags", {}))
    
    # Get version from MLflow tags, fallback to "1.0" if not present
    version_value = tags.get("version", "1.0")

    def format_timestamp(timestamp_ms):
        if timestamp_ms:
            return datetime.fromtimestamp(int(timestamp_ms) / 1000).isoformat()
        return datetime.now().isoformat()

    tmf_json = {
        "id": str(uuid.uuid4()),
        "href": f"https://mycsp.com:8080/tmfapi/serviceCatalogManagement/v4/serviceSpecification/{run_id}",
        "@type": "AIModelSpecification",
        "@baseType": "ServiceSpecification",
        "@schemaLocation": "https://mycsp.com:8080/tmf-api/schema/AIM/AIModelSpecification.schema.json",
        "name": tags.get("mlflow.runName", f"MLflow Run {run_id}"),
        "description": tags.get("description", f"AI Model specification from MLflow run {run_id}"),
        "version": version_value,
        "validFor": {
            "startDateTime": format_timestamp(info.get("start_time")),
            "endDateTime": format_timestamp(info.get("end_time")) if info.get("end_time") else None,
        },
        "lastUpdate": format_timestamp(info.get("end_time") or info.get("start_time")),
        "lifecycleStatus": tags.get("lifecycleStatus", "Active"),
        "isBundle": False,

        # --- TMF 915 Defined Fields ---
        "modelSpecificationHistory": {
            "description": "Model development history and version as preserved in MLflow",
            "url": f"{tracking_uri}/#/experiments/{info.get('experiment_id')}/runs/{run_id}"
        },

        "inheritedModel": {
            "description": "Reference to the base model used (if transfer learning applied)",
            "url": tags.get("inheritedModel_url", "https://link.to.model.repo/modelID=base123")
        },

        "modelTrainingData": {
            "description": "Dataset used to train the model",
            "url": f"{tracking_uri}/#/experiments/{info.get('experiment_id')}/runs/{run_id}/artifacts/dataset"
        },

        "modelEvaluationData": {
            "description": "Evaluation dataset and metrics preserved in MLflow",
            "url": f"{tracking_uri}/#/experiments/{info.get('experiment_id')}/runs/{run_id}/artifacts/evaluation"
        },

        "modelDataSheet": {
            "description": "Digital document describing model characteristics",
            "url": f"{info.get('artifact_uri')}/datasheet",
            "mimeType": "application/json"
        },

        "deploymentRecord": {
            "description": "Deployment approval or rollout record for this model",
            "url": tags.get("deploymentRecord_url", "https://link.to.model.repo/deploymentRecordID=57432.1")
        },

        "modelContractVersionHistory": {
            "description": "Model contract and version history",
            "url": tags.get("contractHistory_url", "https://gitlab.server/modelcontracts/57432/blob/master/contract")
        },

        # --- Characteristics ---
        "serviceSpecCharacteristic": [
            {
                "name": param_key,
                "description": f"Parameter {param_key} from MLflow",
                "valueType": "string",
                "configurable": False,
                "validFor": {
                    "startDateTime": format_timestamp(info.get("start_time")),
                    "endDateTime": format_timestamp(info.get("end_time")),
                },
                "minCardinality": 0,
                "maxCardinality": 1,
                "isUnique": True,
                "regex": "",
                "extensible": False,
                "serviceSpecCharacteristicValue": [
                    {
                        "valueType": "string",
                        "isDefault": True,
                        "value": param_value,
                        "validFor": {
                            "startDateTime": format_timestamp(info.get("start_time")),
                            "endDateTime": format_timestamp(info.get("end_time")),
                        },
                    }
                ],
            }
            for param_key, param_value in params.items()
        ],

        # --- Related Party ---
        "relatedParty": [
            {
                "href": f"https://mycsp.com:8080/tmf-api/partyManagement/v4/individual/{info.get('user_id', 'unknown')}",
                "id": info.get("user_id", "unknown"),
                "name": info.get("user_id", "Unknown"),
                "role": "Supplier"
            }
        ],

        "targetServiceSchema": {
            "@type": "AIModel",
            "@schemaLocation": "https://mycsp.com:8080/tmf-api/schema/AIM/AIModel.schema.json"
        },

        # --- MLflow Metadata (custom extension) ---
        "mlflowMetadata": {
            "runId": run_id,
            "experimentId": info.get("experiment_id"),
            "artifactUri": info.get("artifact_uri"),
            "status": info.get("status"),
            "userId": info.get("user_id"),
            "metrics": metrics,
            "params": params,
            "tags": tags,
        },
    }

    return {k: v for k, v in tmf_json.items() if v is not None}


