import os
import pandas as pd
import networkx as nx
import traceback
from typing import Dict, Any, List
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.readers.file import UnstructuredReader

from graphrag_anthropic_llamaindex.db_manager import (
    calculate_file_hash,
    load_processed_files_db,
    save_processed_files_db,
    load_entities_db,
    save_entities_db,
    load_relationships_db,
    save_relationships_db,
    load_community_db,
    save_community_db,
    load_community_summaries_db,
    save_community_summaries_db,
)
from graphrag_anthropic_llamaindex.file_filter import FileFilter
from graphrag_anthropic_llamaindex.graph_operations import cluster_graph
from graphrag_anthropic_llamaindex.llm_utils import parse_llm_json_output, extraction_prompt_template, summary_prompt_template, _get_full_llm_response_with_continuation
import fsspec
import hashlib
from pathlib import Path



def add_documents(
    input_dir,
    output_dir,
    vector_store=None,
    entity_vector_store=None,
    community_vector_store=None,
    community_detection_config=None,
    use_archive_reader=True,
    file_filter=None,
):
    """Adds documents from the data directory to the index."""
    print(f"Adding documents from '{input_dir}'...")
    
    # Initialize file filter if not provided
    if file_filter is None:
        file_filter = FileFilter()
    
    processed_files_df = load_processed_files_db(output_dir)
    processed_hashes = set(processed_files_df['hash'].tolist())
    newly_processed_files = []

    # Define supported file extensions
    unstructured_supported_exts = [
        ".txt", ".text", ".eml", ".msg", ".html", ".htm", ".xml", ".json",
        ".tsv", ".md", ".rst", ".rtf", ".odt", ".doc", ".docx", ".mermaid", ".plantuml",
        ".ppt", ".pptx", ".pdf", ".png", ".jpg", ".jpeg", ".heic", ".epub",
    ]
    file_extractor = {ext: UnstructuredReader() for ext in unstructured_supported_exts}
    
    # Load documents with unified processing logic
    print("Loading documents...")
    all_documents = _load_documents_with_archives(
        input_dir=input_dir,
        file_extractor=file_extractor,
        show_progress=True,
        file_filter=file_filter,
        use_archive_reader=use_archive_reader
    )
    
    # Process documents and check for duplicates
    for doc in all_documents:
        # Get document source path (virtual or physical)
        source_path = doc.extra_info.get('virtual_path', 
                                        doc.extra_info.get('source_path',
                                                          doc.extra_info.get('file_name', 'unknown')))
        
        # Calculate hash for duplicate detection using SHA-256
        doc_hash = _calculate_document_hash(doc.text, source_path)
        
        if doc_hash in processed_hashes:
            print(f"Skipping already processed document: {source_path}")
            continue
        
        newly_processed_files.append({
            'filepath': source_path,
            'hash': doc_hash,
            'original_path': doc.extra_info.get('source_archive', source_path)
        })
        
        print(f"Added document: {source_path}")
    
    # Filter out already processed documents
    all_documents = [doc for doc in all_documents 
                    if _calculate_document_hash(doc.text, 
                                               doc.extra_info.get('virtual_path', 
                                                                 doc.extra_info.get('source_path',
                                                                                   doc.extra_info.get('file_name', 'unknown')))) 
                    not in processed_hashes]

    if not all_documents:
        print("No new documents to add.")
        return

    try:
        # Chunk documents into nodes
        node_parser = Settings.node_parser
        nodes = node_parser.get_nodes_from_documents(all_documents)

        extracted_entities_list = []
        extracted_relationships_list = []
        entity_documents = [] # For entity vector store

        print("Extracting entities and relationships from document chunks...")
        for i, node in enumerate(nodes):
            try:
                prompt = extraction_prompt_template.format(text=node.text)
                json_output = _get_full_llm_response_with_continuation(prompt)
                result = parse_llm_json_output(json_output)
                
                if result:
                    for entity in result.get('entities', []):
                        extracted_entities_list.append(entity)
                        entity_documents.append(Document(text=entity.get('name', ''), extra_info=entity))
                    for relationship in result.get('relationships', []):
                        extracted_relationships_list.append(relationship)
                print(f"  Processed chunk {i+1}/{len(nodes)}")
            except Exception as e:
                print(f"  Error extracting from chunk {i+1}: {e}")
                traceback.print_exc()
                raise RuntimeError(f"Entity extraction failed for chunk {i+1}") from e

        # Save extracted entities and relationships to Parquet
        if extracted_entities_list:
            new_entities_df = pd.DataFrame(extracted_entities_list)
            existing_entities_df = load_entities_db(output_dir)
            updated_entities_df = pd.concat([existing_entities_df, new_entities_df], ignore_index=True).drop_duplicates(subset=['name', 'type'])
            save_entities_db(updated_entities_df, output_dir)
            print(f"Saved {len(new_entities_df)} new entities to {output_dir}/entities.parquet")

        if extracted_relationships_list:
            new_relationships_df = pd.DataFrame(extracted_relationships_list)
            existing_relationships_df = load_relationships_db(output_dir)
            updated_relationships_df = pd.concat([existing_relationships_df, new_relationships_df], ignore_index=True).drop_duplicates()
            save_relationships_db(updated_relationships_df, output_dir)
            print(f"Saved {len(new_relationships_df)} new relationships to {output_dir}/relationships.parquet")

        # --- Community Detection ---
        if extracted_relationships_list and community_detection_config:
            print("Performing community detection...")
            graph = nx.Graph()
            for rel in extracted_relationships_list:
                graph.add_edge(rel['source'], rel['target'])
            
            max_cluster_size = community_detection_config.get("max_cluster_size", 10)
            use_lcc = community_detection_config.get("use_lcc", True)
            seed = community_detection_config.get("seed", 42)

            communities = cluster_graph(graph, max_cluster_size, use_lcc, seed)
            
            if communities:
                new_communities_df = pd.DataFrame(communities, columns=['level', 'cluster_id', 'parent_cluster', 'nodes'])
                existing_communities_df = load_community_db(output_dir)
                updated_communities_df = pd.concat([existing_communities_df, new_communities_df], ignore_index=True).drop_duplicates(subset=['level', 'cluster_id'])
                save_community_db(updated_communities_df, output_dir)
                print(f"Saved {len(new_communities_df)} new communities to {output_dir}/communities.parquet")

                # --- Community Summarization ---
                print("Generating community summaries...")
                
                extracted_community_summaries = []
                community_summary_documents = [] # For community summary vector store

                entity_to_node_text = {}
                for node in nodes:
                    temp_prompt = extraction_prompt_template.format(text=node.text)
                    temp_json_output = _get_full_llm_response_with_continuation(temp_prompt)
                    temp_result = parse_llm_json_output(temp_json_output)
                    if temp_result:
                        for entity in temp_result.get('entities', []):
                            entity_to_node_text[entity.get('name', '')] = node.text
                
                for community_level, community_id, _, community_nodes in communities:
                    community_text_parts = []
                    key_entities_in_community = []
                    for entity_name in community_nodes:
                        if entity_name in entity_to_node_text:
                            community_text_parts.append(entity_to_node_text[entity_name])
                            key_entities_in_community.append(entity_name)
                    
                    if community_text_parts:
                        combined_community_text = " ".join(community_text_parts)
                        try:
                            prompt = summary_prompt_template.format(text=combined_community_text)
                            json_output = _get_full_llm_response_with_continuation(prompt)
                            summary_dict = parse_llm_json_output(json_output)
                            
                            if summary_dict:
                                summary_dict['community_id'] = community_id # Ensure community_id is set
                                extracted_community_summaries.append(summary_dict)
                                # Flatten metadata for vector store compatibility
                                flat_metadata = {}
                                for key, value in summary_dict.items():
                                    if isinstance(value, (str, int, float, type(None))):
                                        flat_metadata[key] = value
                                    else:
                                        flat_metadata[key] = str(value)
                                community_summary_documents.append(Document(text=summary_dict.get('summary', ''), extra_info=flat_metadata))
                                print(f"  Summarized community {community_id} (Level {community_level})")
                        except Exception as e:
                            print(f"  Error summarizing community {community_id}: {e}")
                            raise RuntimeError(f"Community summarization failed for community {community_id}") from e
            
                if extracted_community_summaries:
                    new_summaries_df = pd.DataFrame(extracted_community_summaries)
                    existing_summaries_df = load_community_summaries_db(output_dir)
                    updated_summaries_df = pd.concat([existing_summaries_df, new_summaries_df], ignore_index=True).drop_duplicates(subset=['community_id'])
                    save_community_summaries_db(updated_summaries_df, output_dir)
                    print(f"Saved {len(new_summaries_df)} new community summaries to {output_dir}/community_summaries.parquet")

                # Create/Update community summary vector index
                if community_summary_documents:
                    try:
                        if community_vector_store:
                            community_storage_context = StorageContext.from_defaults(vector_store=community_vector_store)
                            community_index = VectorStoreIndex(community_summary_documents, storage_context=community_storage_context)
                        else:
                            community_index_dir = os.path.join(output_dir, "community_summaries_index")
                            os.makedirs(community_index_dir, exist_ok=True)
                            community_index = VectorStoreIndex(community_summary_documents)
                            community_index.storage_context.persist(persist_dir=community_index_dir)
                        print("Community summary vector index updated.")
                    except Exception as e:
                        print(f"Error creating community summary vector index: {e}")
                        raise RuntimeError("Community summary vector index creation failed") from e
                else:
                    print("No community summaries extracted for indexing.")
            else:
                print("No communities detected.")
        else:
            print("No relationships extracted for community detection.")

        # Create/Update main text index
        try:
            if vector_store:
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex(nodes, storage_context=storage_context)
            else:
                index = VectorStoreIndex(nodes)
                index.storage_context.persist(persist_dir=output_dir)
            print("Main text index updated.")
        except Exception as e:
            print(f"Error creating main text index: {e}")
            raise RuntimeError("Main text index creation failed") from e

        # Create/Update entity vector index
        if entity_documents:
            try:
                if entity_vector_store:
                    entity_storage_context = StorageContext.from_defaults(vector_store=entity_vector_store)
                    entity_index = VectorStoreIndex(entity_documents, storage_context=entity_storage_context)
                else:
                    # If no specific entity_vector_store, use default storage for entities
                    entity_index_dir = os.path.join(output_dir, "entities_index")
                    os.makedirs(entity_index_dir, exist_ok=True)
                    entity_index = VectorStoreIndex(entity_documents)
                    entity_index.storage_context.persist(persist_dir=entity_index_dir)
                print("Entity vector index updated.")
            except Exception as e:
                print(f"Error creating entity vector index: {e}")
                raise RuntimeError("Entity vector index creation failed") from e
        else:
            print("No entities extracted for indexing.")

        # Update and save the processed files database
        updated_df = pd.concat([processed_files_df, pd.DataFrame(newly_processed_files)], ignore_index=True)
        save_processed_files_db(updated_df, output_dir)

        print("Documents and entities processed successfully.")
    except Exception as e:
        print(f"Error adding documents: {e}")
        traceback.print_exc()
        raise  # Re-raise to prevent silent failures


# Archive processing functions

# Supported archive formats
_SUPPORTED_ARCHIVE_FORMATS = ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2']


def _calculate_document_hash(text: str, source_path: str) -> str:
    """
    Calculate document hash using SHA-256
    
    Args:
        text: Document text content
        source_path: Source path (including virtual paths)
        
    Returns:
        str: SHA-256 hash string
    """
    combined = f"{text}:{source_path}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def _create_archive_filesystem(archive_path: str) -> 'fsspec.AbstractFileSystem':
    """
    Create filesystem for archive file
    
    Args:
        archive_path: Path to archive file
        
    Returns:
        fsspec.AbstractFileSystem: Filesystem for the archive
        
    Raises:
        ValueError: If archive format is not supported
    """
    file_ext = Path(archive_path).suffix.lower()
    
    if file_ext == '.zip':
        return fsspec.filesystem('zip', fo=archive_path)
    elif file_ext in ['.tar', '.tar.gz', '.tgz', '.tar.bz2']:
        return fsspec.filesystem('tar', fo=archive_path)
    else:
        raise ValueError(f"Unsupported archive format: {file_ext} for {archive_path}")


def _find_archive_files(input_dir: str, file_filter: FileFilter = None) -> List[str]:
    """
    Find archive files in directory
    
    Args:
        input_dir: Directory to search
        file_filter: FileFilter instance for filtering files
        
    Returns:
        List[str]: List of archive file paths
    """
    if file_filter is None:
        file_filter = FileFilter()
    
    # Use FileFilter's find_files method with archive extensions
    return file_filter.find_files(input_dir, extensions=_SUPPORTED_ARCHIVE_FORMATS)


def _load_documents_with_archives(
    input_dir: str,
    file_extractor: Dict[str, Any],
    recursive: bool = True,
    show_progress: bool = False,
    file_filter: FileFilter = None,
    use_archive_reader: bool = True
) -> List[Document]:
    """
    Load documents with unified processing logic
    
    Args:
        input_dir: Input directory path
        file_extractor: File extractor mapping
        recursive: Whether to search recursively
        show_progress: Whether to show progress
        file_filter: FileFilter instance for filtering files
        use_archive_reader: Whether to process archive files
        
    Returns:
        List[Document]: Loaded documents
    """
    all_docs = []
    
    # Initialize file filter if not provided
    if file_filter is None:
        file_filter = FileFilter()
    
    # Process regular files
    all_docs.extend(_process_regular_files(input_dir, file_extractor, recursive, show_progress, file_filter))
    
    # Process archive files only if enabled
    if use_archive_reader:
        archive_files = _find_archive_files(input_dir, file_filter)
        for archive_path in archive_files:
            all_docs.extend(_process_archive_files(archive_path, file_extractor, show_progress, file_filter))
    
    return all_docs


def _process_regular_files(
    input_dir: str,
    file_extractor: Dict[str, Any],
    recursive: bool,
    show_progress: bool,
    file_filter: FileFilter
) -> List[Document]:
    """Process regular files with CSV special handling"""
    all_docs = []
    
    # Find all files
    all_file_paths = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                all_file_paths.append(file_path)
    else:
        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path):
                all_file_paths.append(file_path)
    
    # Filter files
    all_file_paths = file_filter.filter_file_paths(all_file_paths)
    
    # Separate CSV and non-CSV files
    csv_files = [f for f in all_file_paths if f.endswith('.csv')]
    non_csv_files = [f for f in all_file_paths if not f.endswith('.csv')]
    
    # Process CSV files with row-by-row logic
    for csv_path in csv_files:
        if show_progress:
            print(f"Processing CSV file: {csv_path}")
        all_docs.extend(_process_csv_file(csv_path))
    
    # Process non-CSV files with UnstructuredReader
    if non_csv_files:
        reader = SimpleDirectoryReader(
            input_files=non_csv_files,
            file_extractor=file_extractor,
            recursive=False
        )
        non_csv_docs = reader.load_data(show_progress=show_progress)
        all_docs.extend(non_csv_docs)
    
    return all_docs


def _process_archive_files(
    archive_path: str,
    file_extractor: Dict[str, Any],
    show_progress: bool,
    file_filter: FileFilter
) -> List[Document]:
    """Process archive files with CSV special handling"""
    all_docs = []
    
    if show_progress:
        print(f"Processing archive: {archive_path}")
    
    try:
        archive_fs = _create_archive_filesystem(archive_path)
        
        # List all files in archive
        all_archive_files = archive_fs.find("", detail=False)
        
        # Separate CSV and non-CSV files
        csv_files = [f for f in all_archive_files if f.endswith('.csv')]
        non_csv_files = [f for f in all_archive_files if not f.endswith('.csv')]
        
        # Process CSV files from archive
        for csv_file in csv_files:
            if show_progress:
                print(f"Processing CSV from archive: {csv_file}")
            all_docs.extend(_process_csv_from_archive(csv_file, archive_path, archive_fs))
        
        # Process non-CSV files from archive
        if non_csv_files:
            # Filter non-CSV files from archive
            filtered_non_csv_files = file_filter.filter_file_paths(non_csv_files)
            if filtered_non_csv_files:
                archive_reader = SimpleDirectoryReader(
                    input_dir="",
                    fs=archive_fs,
                    file_extractor=file_extractor,
                    file_metadata=lambda fname: _create_archive_metadata(fname, archive_path)
                )
                archive_docs = archive_reader.load_data(show_progress=show_progress)
                all_docs.extend(archive_docs)
        
        if show_progress:
            print(f"Loaded {len(all_docs)} documents from archive: {archive_path}")
    
    except Exception as e:
        print(f"Error processing archive {archive_path}: {e}")
        raise RuntimeError(f"Archive processing failed: {archive_path}") from e
    
    return all_docs

def _process_csv_file(csv_path: str) -> List[Document]:
    """Process CSV file with row-by-row document creation"""
    documents = []
    try:
        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            doc_content = ", ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(Document(
                text=doc_content,
                extra_info={
                    "file_name": os.path.basename(csv_path),
                    "row_index": index,
                    "source_path": csv_path
                }
            ))
    except Exception as e:
        print(f"Error processing CSV file {csv_path}: {e}")
        raise RuntimeError(f"CSV processing failed: {csv_path}") from e
    
    return documents


def _process_csv_from_archive(csv_file: str, archive_path: str, archive_fs) -> List[Document]:
    """Process CSV file from archive with row-by-row document creation"""
    documents = []
    try:
        with archive_fs.open(csv_file, 'r') as f:
            df = pd.read_csv(f)
            for index, row in df.iterrows():
                doc_content = ", ".join([f"{col}: {val}" for col, val in row.items()])
                documents.append(Document(
                    text=doc_content,
                    extra_info={
                        "file_name": os.path.basename(csv_file),
                        "row_index": index,
                        "source_archive": archive_path,
                        "archive_internal_path": csv_file,
                        "virtual_path": f"{archive_path}!/{csv_file}",
                        "is_from_archive": True
                    }
                ))
    except Exception as e:
        print(f"Error processing CSV from archive {csv_file}: {e}")
        raise RuntimeError(f"Archive CSV processing failed: {csv_file}") from e
    
    return documents


def _create_archive_metadata(internal_path: str, archive_path: str) -> Dict[str, Any]:
    """
    Create metadata for archive file
    
    Args:
        internal_path: Path within archive
        archive_path: Path to archive file
        
    Returns:
        Dict[str, Any]: Metadata dictionary
    """
    return {
        "source_archive": archive_path,
        "archive_internal_path": internal_path,
        "virtual_path": f"{archive_path}!/{internal_path}",
        "is_from_archive": True
    }

