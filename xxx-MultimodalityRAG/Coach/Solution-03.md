# Challenge 03 - Indexing with AI Search - Coach's Guide 

[< Previous Solution](./Solution-02.md) - **[Home](./README.md)** - [Next Solution >](./Solution-04.md)

## Notes & Guidance
In this section, we will run the indexer to create an index. There are also a few examples of queries to test the index as well.

### Run the Indexer
Using the data in the connected blob storage container, the indexer will create an index with the specified configuration using the previously defined skillsets
```
def create_and_run_indexer(indexer_client, indexer_name, skillset_name, index_name, data_source_name):
    indexer = SearchIndexer(
        name=indexer_name,
        description="Indexer to index images and generate embeddings",
        skillset_name=skillset_name,
        target_index_name=index_name,
        data_source_name=data_source_name,
        parameters=IndexingParameters(
            configuration=IndexingParametersConfiguration(
                data_to_extract= "contentAndMetadata",
                parsing_mode="delimitedText",
                first_line_contains_headers= True,
                delimited_text_delimiter=",",
                delimited_text_headers="",
                query_timeout=None,
            ),
        ),
        field_mappings=[FieldMapping(source_field_name="IssueID", target_field_name="IssueID")],
        output_field_mappings=[
            FieldMapping(source_field_name="/document/imageVector", target_field_name="imageVector"),
        ],
    )

    indexer_client.create_or_update_indexer(indexer)
    print(f"{indexer_name} created or updated.")

    indexer_client.run_indexer(indexer_name)
    print(f"{indexer_name} is running. If queries return no results, please wait a bit and try again.")

indexer_client = SearchIndexerClient(
    endpoint=SEARCH_SERVICE_ENDPOINT, credential=AZURE_SEARCH_CREDENTIAL
)
data_source_name = f"{BLOB_CONTAINER_NAME}-blob"
indexer_name = f"{INDEX_NAME}-indexer"

create_and_run_indexer(indexer_client, indexer_name, skillset_name, INDEX_NAME, data_source_name)
```

### Run Example Query
Below is an example of how to create a Search client and then run a search query. Further, it will output the results in an organized and detailed manner. 
```
# Initialize the SearchClient
search_client = SearchClient(
    SEARCH_SERVICE_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AZURE_SEARCH_CREDENTIAL,
)

# Define the query
query = "altman" # English Query

vector_query = VectorizableTextQuery(
    text=query,
    k_nearest_neighbors=3,
    #fields="captionVector",
    fields="imageVector",
)

# Perform the search
results = search_client.search(
    search_text=None,
    vector_queries=[vector_query],
    select=["IssueID", "Bipad_Title", "Profit", "Content_Description","imageUrl"],
    top=3
)

# Print the results
for result in results:
    print(f"Score: {result['@search.score']}")
    display(HTML(f'<img src="{result["imageUrl"]}" style="width:200px;"/>'))
    print(f"IssueID: {result['IssueID']}")  
    print(f"Bipad_Title: {result['Bipad_Title']}")  
    print(f"Content_Description: {result['Content_Description']}")
    print(f"Profit: {result['Profit']}")
    print("-" * 50) 
```
