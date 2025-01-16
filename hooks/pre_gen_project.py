import glob
import json
import sys
from typing import List, Literal, Optional, Union

# Importamos clases y tipos necesarios de Pydantic
from pydantic import BaseModel, Field, ValidationError, conint, constr

# --------------------------------------------------------------------
# Definiciones de tipos reutilizables
# --------------------------------------------------------------------
NamePatternStr = constr(
    min_length=3, max_length=30, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$"
)

EmailStr = constr(min_length=6, max_length=254, pattern=r"^[^@]+@[^@]+\.[^@]+$")

ServiceAccountName = constr(
    min_length=6, max_length=30, pattern=r"^[a-z][a-z0-9\-]{4,28}[a-z0-9]$"
)

BucketName = constr(
    min_length=3, max_length=63, pattern=r"^[a-z0-9][a-z0-9\-_.]{1,61}[a-z0-9]$"
)

DatasetName = constr(min_length=1, max_length=1024, pattern=r"^[a-zA-Z0-9_]{1,1024}$")


# --------------------------------------------------------------------
# Modelos Pydantic
# --------------------------------------------------------------------
class MLPipeline(BaseModel):
    """
    Define la configuración de un pipeline de Machine Learning.
    """

    # 1-2) Project y Application
    projectName: NamePatternStr = Field(
        ...,
        description="Name of the project, identifying a specific problem. Known across the company.",
    )
    applicationName: NamePatternStr = Field(
        ...,
        description="Name of the application for a specific objective. Can be known internally or company-wide.",
    )

    # 3) Descripción
    projectDescription: constr(min_length=3, max_length=250) = Field(
        ...,
        description="Description of the application/project and its relevance within the company.",
    )

    # 4-5) Cuentas Admin y Viewer
    adminAccounts: List[EmailStr] = Field(
        ...,
        description="List of valid email addresses with Admin access for created resources.",
    )
    viewerAccounts: List[EmailStr] = Field(
        ...,
        description="List of valid email addresses with Viewer access for created resources.",
    )

    # 6-8) Service Accounts
    serviceAccountMaasName: Optional[List[ServiceAccountName]] = Field(
        None, description="Service account(s) for the GCP deployment project."
    )
    serviceAccountExplorationName: Optional[List[ServiceAccountName]] = Field(
        None, description="Service account(s) for the GCP exploration project."
    )
    serviceAccountDiscoveryName: Optional[List[ServiceAccountName]] = Field(
        None, description="Service account(s) for the GCP discovery project."
    )

    # 9-11) Buckets
    bucketMaasName: Optional[List[BucketName]] = Field(
        None,
        description="Name(s) of Cloud Storage bucket(s) for the GCP deployment project.",
    )
    bucketExplorationName: Optional[BucketName] = Field(
        None,
        description="Name of the Cloud Storage bucket for the GCP exploration project.",
    )
    bucketDiscoveryName: Optional[BucketName] = Field(
        None,
        description="Name of the Cloud Storage bucket for the GCP discovery project.",
    )

    # 12-14) Datasets
    datasetMaasName: Optional[List[DatasetName]] = Field(
        None,
        description="Name(s) of BigQuery dataset(s) for the GCP deployment project.",
    )
    datasetExplorationName: Optional[DatasetName] = Field(
        None,
        description="Name of the BigQuery dataset for the GCP exploration project.",
    )
    datasetDiscoveryName: Optional[DatasetName] = Field(
        None, description="Name of the BigQuery dataset for the GCP discovery project."
    )

    # 15-18) Recursos de Cómputo
    ComputeResourcesCPU: conint(ge=1, le=96) = Field(
        ...,
        description="Number of vCPUs (1-96). Based on Compute Engine and Vertex AI limits.",
    )
    ComputeResourcesRAM: conint(ge=1, le=624) = Field(
        ...,
        description="RAM in GB (1-624). Based on Compute Engine and Vertex AI limits.",
    )
    ComputeResourcesStorage: conint(ge=10, le=65536) = Field(
        ..., description="Storage in GB (10-65536). Based on Compute Engine limits."
    )
    ComputeResourcesGPU: conint(ge=0, le=8) = Field(
        ...,
        description="Number of GPUs (0-8). Based on Vertex AI Training and Workbench limits.",
    )

    # 19) Fuentes de Datos
    Sources: List[constr(min_length=3, max_length=200)] = Field(
        ..., description="Paths of data sources (Cloud Storage, BigQuery, etc.)."
    )

    # 20) Tipo de Modelo
    ModelType: List[
        Literal["classification", "regression", "clustering", "gen-ai", "time-series"]
    ] = Field(
        ...,
        description="Type(s) of model: classification, regression, clustering, gen-ai or time-series.",
    )

    # 21) Esquema de Inferencia
    InferenceSchema: dict = Field(
        ..., description="Schema for model inference outputs."
    )

    # 22-23) Tamaño Estimado
    SourceSizeEstimationKB: conint(ge=1, le=1_000_000_000) = Field(
        ..., description="Estimated size of data sources in KB (1 KB to 1 TB)."
    )
    ModelSizeEstimationKB: conint(ge=1, le=1_000_000_000) = Field(
        ..., description="Estimated size of the model(s) in KB (1 KB to 1 TB)."
    )

    # 24) Runtime Base
    runtimeBase: Literal[
        "Python3.9", "Python3.10", "ApacheBeam", "R", "Dataproc", "TF", "Pytorch"
    ] = Field(
        ..., description="Runtime for installing dependencies and running pipelines."
    )


class AgenticPipeline(BaseModel):
    """
    Define la configuración de un pipeline tipo 'Agentic'.
    """

    # 1-2) Project y Application
    projectName: NamePatternStr = Field(
        ...,
        description="Name of the project, identifying a specific problem. Known across the company.",
    )
    applicationName: NamePatternStr = Field(
        ...,
        description="Name of the application for a specific objective. Can be known internally or company-wide.",
    )

    # 3) Descripción
    projectDescription: constr(min_length=3, max_length=250) = Field(
        ...,
        description="Description of the application/project and its relevance within the company.",
    )

    # 4-5) Cuentas Admin y Viewer
    adminAccounts: List[EmailStr] = Field(
        ...,
        description="List of valid email addresses with Admin access for created resources.",
    )
    viewerAccounts: List[EmailStr] = Field(
        ...,
        description="List of valid email addresses with Viewer access for created resources.",
    )

    # 6-8) Service Accounts
    serviceAccountMaasName: Optional[List[ServiceAccountName]] = Field(
        None, description="Service account(s) for the GCP deployment project."
    )
    serviceAccountExplorationName: Optional[List[ServiceAccountName]] = Field(
        None, description="Service account(s) for the GCP exploration project."
    )
    serviceAccountDiscoveryName: Optional[List[ServiceAccountName]] = Field(
        None, description="Service account(s) for the GCP discovery project."
    )

    # 9-11) Buckets
    bucketMaasName: Optional[List[BucketName]] = Field(
        None,
        description="Name(s) of Cloud Storage bucket(s) for the GCP deployment project.",
    )
    bucketExplorationName: Optional[BucketName] = Field(
        None,
        description="Name of the Cloud Storage bucket for the GCP exploration project.",
    )
    bucketDiscoveryName: Optional[BucketName] = Field(
        None,
        description="Name of the Cloud Storage bucket for the GCP discovery project.",
    )

    # 12-14) Datasets
    datasetMaasName: Optional[List[DatasetName]] = Field(
        None,
        description="Name(s) of BigQuery dataset(s) for the GCP deployment project.",
    )
    datasetExplorationName: Optional[DatasetName] = Field(
        None,
        description="Name of the BigQuery dataset for the GCP exploration project.",
    )
    datasetDiscoveryName: Optional[DatasetName] = Field(
        None, description="Name of the BigQuery dataset for the GCP discovery project."
    )

    # 15-18) Recursos de Cómputo
    ComputeResourcesCPU: conint(ge=1, le=96) = Field(
        ...,
        description="Number of vCPUs (1-96). Based on Compute Engine and Vertex AI limits.",
    )
    ComputeResourcesRAM: conint(ge=1, le=624) = Field(
        ...,
        description="RAM in GB (1-624). Based on Compute Engine and Vertex AI limits.",
    )
    ComputeResourcesStorage: conint(ge=10, le=65536) = Field(
        ..., description="Storage in GB (10-65536). Based on Compute Engine limits."
    )
    ComputeResourcesGPU: conint(ge=0, le=8) = Field(
        ...,
        description="Number of GPUs (0-8). Based on Vertex AI Training and Workbench limits.",
    )

    # 19) Fuentes de Datos
    Sources: List[constr(min_length=3, max_length=200)] = Field(
        ..., description="Paths of data sources (Cloud Storage, BigQuery, etc.)."
    )

    # 21) Esquema de Inferencia
    InferenceSchema: dict = Field(
        ..., description="Schema for model inference outputs."
    )

    # 22-23) Tamaño Estimado
    SourceSizeEstimationKB: conint(ge=1, le=1_000_000_000) = Field(
        ..., description="Estimated size of data sources in KB (1 KB to 1 TB)."
    )

    # 24) Runtime Base
    runtimeBase: Literal[
        "Python3.9", "Python3.10", "ApacheBeam", "R", "Dataproc", "TF", "Pytorch"
    ] = Field(
        ..., description="Runtime for installing dependencies and running pipelines."
    )


# --------------------------------------------------------------------
# Unión de modelos (MLPipeline o AgenticPipeline)
# --------------------------------------------------------------------
JSONFormat = Union[MLPipeline, AgenticPipeline]


# --------------------------------------------------------------------
# Función principal
# --------------------------------------------------------------------
def main() -> int:
    """
    Valida todos los archivos .json en el directorio actual y sus subdirectorios
    contra los modelos Pydantic definidos. Si alguno falla, imprime el error y
    finaliza con código de salida 1; si todos pasan, finaliza con 0.
    """
    # Se buscan todos los archivos .json del repositorio
    json_files = glob.glob("../cookiecutter-config/*.json", recursive=True)

    if not json_files:
        print("[INFO] No se encontraron archivos JSON para validar.")
        return 0

    for f in json_files:
        with open(f, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"[ERROR] El archivo {f} no es un JSON válido.")
                return 1

            try:
                # Pydantic intentará parsear 'data' como MLPipeline o AgenticPipeline
                JSONFormat.parse_obj(data)
            except ValidationError as ve:
                print(f"[ERROR] El archivo {f} no cumple con el modelo Pydantic:")
                # Muestra mensajes detallados de la validación
                print(ve.json(indent=2))
                return 1

    print("✓ Todos los archivos cumplen con los modelos definidos en Pydantic.")
    return 0


if __name__ == "__main__":
    # Utilizamos sys.exit para devolver el código de salida apropiado.
    sys.exit(main())
