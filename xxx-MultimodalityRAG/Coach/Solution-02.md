# Challenge 02 - Data Processing - Coach's Guide 

[< Previous Solution](./Solution-01.md) - **[Home](./README.md)** - [Next Solution >](./Solution-03.md)

## Notes & Guidance

In this section of the Hack, students will create a Jupyter Notebook in Azure Machine Learning Studio using their AML reosurce. Ensure that students are able to properly set this enviornment up. This includes creating a compute instance as well. 

Below are the snippets of code of how to set up the solution.

### Install Libraries

```
import sys
!{sys.executable} -m pip install azure-search-documents --pre --quiet
!{sys.executable} -m  pip install openai python-dotenv azure-identity cohere azure-ai-vision-imageanalysis --quiet

```

We will need this version of azure search documents to ensure the proper running of the code

```
pip install azure-search-documents==11.6.0b4
```
```
pip show azure-search-documents
```

### Set Up Parameters
Here we gather all resource keys and endpoints we created in the last section of the hack.
```
import json
from uuid import uuid4

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

from azure.core.credentials import AzureKeyCredential
import azure.identity #import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AIServicesVisionParameters,
    AIServicesVisionVectorizer,
    AIStudioModelCatalogName,
    AzureMachineLearningVectorizer,
    AzureOpenAIVectorizer,
    AzureOpenAIModelName,
    AzureOpenAIParameters,
    BlobIndexerDataToExtract,
    BlobIndexerParsingMode,
    CognitiveServicesAccountKey,
    DefaultCognitiveServicesAccount,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    FieldMapping,
    HnswAlgorithmConfiguration,
    HnswParameters,
    IndexerExecutionStatus,
    IndexingParameters,
    IndexingParametersConfiguration,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    ScalarQuantizationCompressionConfiguration,
    ScalarQuantizationParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataIdentity,
    SearchIndexerDataSourceConnection,
    SearchIndexerSkillset,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
    VisionVectorizeSkill,
)
from azure.search.documents.models import (
    HybridCountAndFacetMode,
    HybridSearch,
    SearchScoreThreshold,
    VectorizableTextQuery,
    VectorizableImageBinaryQuery,
    VectorizableImageUrlQuery,
    VectorSimilarityThreshold,
)
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from IPython.display import Image, display, HTML
from openai import AzureOpenAI

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv("globalparameters.env"))

# Get environment variables for Azure AI Vision
try:
    # Azure Blob Storage credentials  
    STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
    STORAGE_ACCOUNT_KEY = os.getenv("STORAGE_ACCOUNT_KEY")
    BLOB_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"  
    BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
    # Configuration AI Vision and Search should be in the same region (EAST US)
    AZURE_AI_VISION_API_KEY = os.getenv("AI_SERVICE_KEY")
    AZURE_AI_VISION_ENDPOINT = os.getenv("AI_SERVICE_ENDPOINT")
    # Configuration of Azure Search
    INDEX_NAME = os.getenv("INDEX_NAME")
    SEARCH_SERVICE_API_KEY = os.getenv("SEARCH_SERVICE_API_KEY") #Admin Key
    SEARCH_SERVICE_ENDPOINT = os.getenv("SEARCH_SERVICE_ENDPOINT")
    #Configuration of openAI
    AZURE_OPENAI_VERSION=os.getenv('AZURE_OPENAI_VERSION'),
    AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT'),
    AZURE_OPENAI_KEY=os.getenv('AZURE_OPENAI_KEY'),

except KeyError as e:
    print(f"Missing environment variable: {str(e)}")
    print("Set them before running this sample.")
    exit()
```

### Get Azure Search Credential
Ensures Azure Search can be accessed using students' perfered authentication method. 
```
# User-specified parameter
USE_AAD_FOR_SEARCH = False  # Set this to False to use API key for authentication

def authenticate_azure_search(api_key=None, use_aad_for_search=False):
    if use_aad_for_search:
        print("Using AAD for authentication.")
        credential = DefaultAzureCredential()
    else:
        print("Using API keys for authentication.")
        if api_key is None:
            raise ValueError("API key must be provided if not using AAD for authentication.")
        credential = AzureKeyCredential(api_key)
    return credential

AZURE_SEARCH_CREDENTIAL = authenticate_azure_search(api_key=SEARCH_SERVICE_API_KEY, use_aad_for_search=USE_AAD_FOR_SEARCH)
```

### Create blob data source connector on AI Search Indexer
This will enable our AI Search resource to be able to communicate with Blob Storage. Thus enabling the stored data/images to be leveraged to create indexes. 
```
def create_or_update_data_source(indexer_client, container_name, connection_string, index_name):
    """
    Create or update a data source connection for Azure AI Search.
    """
    container = SearchIndexerDataContainer(name=container_name)
    print(container)
    data_source_connection = SearchIndexerDataSourceConnection(
        name=f"{index_name}-blob",
        type="azureblob",
        connection_string=connection_string,
        container=container
    )
    print(data_source_connection)
    try:
        indexer_client.create_or_update_data_source_connection(data_source_connection)
        print(f"Data source '{index_name}-blob' created or updated successfully.")
    except Exception as e:
        raise Exception(f"Failed to create or update data source due to error: {e}")

# Create a SearchIndexerClient instance
indexer_client = SearchIndexerClient(SEARCH_SERVICE_ENDPOINT, AZURE_SEARCH_CREDENTIAL)

# Call the function to create or update the data source
create_or_update_data_source(indexer_client, BLOB_CONTAINER_NAME, BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME)

```

### Create Search Index
Creating an Index with the proper fields and configurations.
```
def create_fields():
    """Creates the fields for the search index based on the specified schema."""
    return [
        #SearchField(name="AzureSearch_DocumentKey",  key=True, type=SearchFieldDataType.String),
        SimpleField(name="IssueID", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchField(name="imageUrl", type=SearchFieldDataType.String, searchable=True),      
        SearchField(name="Bipad_Title", type=SearchFieldDataType.String, filterable=True),  
        SearchField(name="On_Sale_Date", type=SearchFieldDataType.DateTimeOffset, sortable=True, filterable=True, facetable=False),   
        SearchField(name="Magazine_Category_Description", type=SearchFieldDataType.String, sortable=False, filterable=True, facetable=False),
        SearchField(name="Magazine_frequency", type=SearchFieldDataType.String, sortable=False, filterable=True, facetable=False),
        SearchField(name="Content_Description", type=SearchFieldDataType.String, sortable=False, filterable=False, facetable=False),
        SearchField(name="TotalORDraw", type=SearchFieldDataType.Int32, sortable=True, filterable=True, facetable=False),
        SearchField(name="BillingDraw", type=SearchFieldDataType.Int32, sortable=True, filterable=True, facetable=False),
        SearchField(name="POS_Sale", type=SearchFieldDataType.Int32, sortable=True, filterable=True, facetable=False),
        SearchField(name="Status", type=SearchFieldDataType.String, sortable=False, filterable=True, facetable=False),
        SearchField(name="Profit", type=SearchFieldDataType.Int32, sortable=True, filterable=True, facetable=False),
        SearchField(
            name="imageVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1024,
            vector_search_profile_name="myHnswProfile",
            stored=False,
        ),
    ]

def create_vector_search_configuration():
    """Creates the vector search configuration."""
    return VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )
        ],
        compressions=[
            ScalarQuantizationCompressionConfiguration(
                name="myScalarQuantization",
                compression_name="scalar",
                rerank_with_original_vectors=True,
                default_oversampling=10,
                parameters=ScalarQuantizationParameters(quantized_data_type="int8"),
            )
        ],
        vectorizers=[
            AIServicesVisionVectorizer(
                name="myAIServicesVectorizer",
                kind="aiServicesVision",
                ai_services_vision_parameters=AIServicesVisionParameters(
                    model_version="2023-04-15",#2023-04-15
                    resource_uri=AZURE_AI_VISION_ENDPOINT,
                    api_key=AZURE_AI_VISION_API_KEY,
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
                compression_configuration_name="myScalarQuantization",
                vectorizer="myAIServicesVectorizer",
            )
        ],
    )


def create_search_index(index_client, index_name, fields, vector_search):
    """Creates or updates a search index."""
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )
    index_client.create_or_update_index(index=index)


index_client = SearchIndexClient(
    endpoint=SEARCH_SERVICE_ENDPOINT, credential=AZURE_SEARCH_CREDENTIAL
)
fields = create_fields()
vector_search = create_vector_search_configuration()

# Create the search index with the adjusted schema
create_search_index(index_client, INDEX_NAME, fields, vector_search)
print(f"Created index: {INDEX_NAME}")
```

### Create Skillsets
The created skillset will create embeddings for images to later be searched.
```
def create_image_embedding_skill():
    return VisionVectorizeSkill(
        name="image-embedding-skill",
        description="Skill to generate embeddings for image via Azure AI Vision",
        context="/document",
        model_version="2023-04-15",#2023-04-15
        inputs=[InputFieldMappingEntry(name="url", source="/document/imageUrl")],
        outputs=[OutputFieldMappingEntry(name="vector", target_name="imageVector")],
        )

def create_skillset(client, skillset_name, image_embedding_skill):
    skillset = SearchIndexerSkillset(
        name=skillset_name,
        description="Skillset for generating embeddings",
        skills=[image_embedding_skill],
        cognitive_services_account=CognitiveServicesAccountKey(key=AZURE_AI_VISION_API_KEY),
    )
    client.create_or_update_skillset(skillset)

client = SearchIndexerClient(
    endpoint=SEARCH_SERVICE_ENDPOINT, credential= AZURE_SEARCH_CREDENTIAL)
skillset_name = f"{INDEX_NAME}-skillset"
image_embedding_skill = create_image_embedding_skill()

create_skillset(client, skillset_name, image_embedding_skill)
print(f"Created skillset: {skillset_name}")
```


