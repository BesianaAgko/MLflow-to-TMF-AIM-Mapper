from mlflow.tracking import MlflowClient
from datetime import datetime
import uuid

def dynamic_mlflow_to_tmf(run_id):
    client = MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics
    tags = run.data.tags

    # Δυναμικά sections
    tag_sections = {}
    for tag_key, tag_value in tags.items():
        if "_" in tag_key:
            prefix = tag_key.split("_")[0]
            clean_key = tag_key.replace(f"{prefix}_", "")
            if prefix not in tag_sections:
                tag_sections[prefix] = {}
            tag_sections[prefix][clean_key] = tag_value

    # Helper function για timestamps
    def format_timestamp(timestamp_ms):
        if timestamp_ms:
            return datetime.fromtimestamp(timestamp_ms / 1000).isoformat()
        return datetime.now().isoformat()

    tmf_json = {
        "id": str(uuid.uuid4()),
        "href": f"https://mycsp.com:8080/tmfapi/serviceCatalogManagement/v4/serviceSpecification/{run_id}",
        "@type": "AIModelSpecification",
        "@baseType": "ServiceSpecification",
        "@schemaLocation": "https://mycsp.com:8080/tmf-api/schema/AIM/AIModelSpecification.schema.json",
        "name": tags.get("mlflow.runName", f"MLflow Run {run_id}"),
        "description": tags.get("description", f"AI Model specification from MLflow run {run_id}"),
        "version": tags.get("version", "1.0"),
        "validFor": {
            "startDateTime": format_timestamp(run.info.start_time),
            "endDateTime": format_timestamp(run.info.end_time) if run.info.end_time else None
        },
        "lastUpdate": format_timestamp(run.info.end_time or run.info.start_time),
        "lifecycleStatus": tags.get("lifecycleStatus", "Active"),
        "isBundle": tags.get("isBundle", "false").lower() == "true",
        
        # Model-specific TMF fields
        "modelSpecificationHistory": {
            "description": "Model development history preserved in MLflow",
            "url": f"{tags.get('mlflow_tracking_uri', 'http://localhost:5000')}/#/experiments/{run.info.experiment_id}/runs/{run_id}"
        },
        "inheritedModel": {
            "description": tags.get("inheritedModel_description", "Reference to parent model used via transfer learning"),
            "url": tags.get("inheritedModel_url", "")
        } if tags.get("inheritedModel_url") else None,
        "modelTrainingData": {
            "description": tags.get("trainingData_description", "Repository link for training data"),
            "url": tags.get("training_dataSource", "")
        } if tags.get("training_dataSource") else None,
        "modelEvaluationData": {
            "description": tags.get("evaluationData_description", "Repository link for evaluation data"),
            "url": tags.get("evaluation_dataSource", "")
        } if tags.get("evaluation_dataSource") else None,
        "modelDataSheet": {
            "description": tags.get("dataSheet_description", "Digital document describing this model"),
            "url": tags.get("dataSheet_url", ""),
            "mimeType": tags.get("dataSheet_mimeType", "application/json")
        } if tags.get("dataSheet_url") else None,
        "deploymentRecord": {
            "description": tags.get("deploymentRecord_description", "Deployment approval record for this model"),
            "url": tags.get("deploymentRecord_url", "")
        } if tags.get("deploymentRecord_url") else None,
        "modelContractVersionHistory": {
            "description": tags.get("contractHistory_description", "Model contract and version history"),
            "url": tags.get("contractHistory_url", "")
        } if tags.get("contractHistory_url") else None,

        # Service characteristics από MLflow parameters
        "serviceSpecCharacteristic": [
            {
                "name": param_key,
                "description": tags.get(f"{param_key}_description", f"Parameter {param_key} from MLflow"),
                "valueType": tags.get(f"{param_key}_valueType", "string"),
                "configurable": tags.get(f"{param_key}_configurable", "false").lower() == "true",
                "validFor": {
                    "startDateTime": format_timestamp(run.info.start_time),
                    "endDateTime": format_timestamp(run.info.end_time) if run.info.end_time else None
                },
                "minCardinality": int(tags.get(f"{param_key}_minCardinality", "0")),
                "maxCardinality": int(tags.get(f"{param_key}_maxCardinality", "1")),
                "isUnique": tags.get(f"{param_key}_isUnique", "true").lower() == "true",
                "regex": tags.get(f"{param_key}_regex", ""),
                "extensible": tags.get(f"{param_key}_extensible", "false").lower() == "true",
                "serviceSpecCharacteristicValue": [
                    {
                        "valueType": tags.get(f"{param_key}_valueType", "string"),
                        "isDefault": True,
                        "value": param_value,
                        "validFor": {
                            "startDateTime": format_timestamp(run.info.start_time),
                            "endDateTime": format_timestamp(run.info.end_time) if run.info.end_time else None
                        }
                    }
                ]
            } for param_key, param_value in params.items()
        ],

        # Related parties
        "relatedParty": [
            {
                "href": tags.get("owner_href", f"https://mycsp.com:8080/tmf-api/partyManagement/v4/individual/{tags.get('owner_id', 'unknown')}"),
                "id": tags.get("owner_id", "unknown"),
                "name": tags.get("owner_name", run.info.user_id or "Unknown"),
                "role": tags.get("owner_role", "ModelOwner")
            }
        ],

        # Target service schema
        "targetServiceSchema": {
            "@type": "AIModel",
            "@schemaLocation": "https://mycsp.com:8080/tmf-api/schema/AIM/AIModel.schema.json"
        },

        # MLflow specific metadata (custom extension)
        "mlflowMetadata": {
            "runId": run_id,
            "experimentId": run.info.experiment_id,
            "artifactUri": run.info.artifact_uri,
            "status": run.info.status,
            "userId": run.info.user_id,
            "metrics": metrics,
            "allTags": tags
        }
    }
    
   
    return {k: v for k, v in tmf_json.items() if v is not None}

# Test function
def test_conversion():
    
    run_id = "your_run_id_here"
    result = dynamic_mlflow_to_tmf(run_id)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_conversion()

