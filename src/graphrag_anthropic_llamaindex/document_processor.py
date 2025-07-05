import os
import pandas as pd
import networkx as nx
import traceback
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
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
from graphrag_anthropic_llamaindex.graph_operations import cluster_graph
from graphrag_anthropic_llamaindex.llm_utils import parse_llm_json_output, extraction_prompt_template, summary_prompt_template

def add_documents(
    input_dir,
    output_dir,
    vector_store=None,
    entity_vector_store=None,
    community_vector_store=None,
    service_context=None,
    community_detection_config=None,
):
    """Adds documents from the data directory to the index."""
    print(f"Adding documents from '{input_dir}'...")
    try:
        processed_files_df = load_processed_files_db(output_dir)
        processed_hashes = set(processed_files_df['hash'].tolist())
        newly_processed_files = []

        all_documents = []

        # Get all files in the data directory, including subdirectories
        all_file_paths = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                all_file_paths.append(os.path.join(root, file))

        # Separate CSV and non-CSV files
        csv_file_paths = [f for f in all_file_paths if f.endswith('.csv')]
        non_csv_file_paths = [f for f in all_file_paths if not f.endswith('.csv')]

        # Process CSV files
        for csv_path in csv_file_paths:
            file_hash = calculate_file_hash(csv_path)
            if file_hash in processed_hashes:
                print(f"Skipping already processed CSV file: {csv_path}")
                continue

            print(f"Processing CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            for index, row in df.iterrows():
                doc_content = ", ".join([f"{col}: {val}" for col, val in row.items()])
                all_documents.append(Document(text=doc_content, extra_info={"file_name": os.path.basename(csv_path), "row_index": index}))
            newly_processed_files.append({'filepath': csv_path, 'hash': file_hash})

        # Process non-CSV files using UnstructuredReader
        unstructured_supported_exts = [
            ".txt", ".text", ".eml", ".msg", ".html", ".htm", ".xml", ".json",
            ".tsv", ".md", ".rst", ".rtf", ".odt", ".doc", ".docx",
            ".ppt", ".pptx", ".pdf", ".png", ".jpg", ".jpeg", ".heic", ".epub",
        ]
        file_extractor = {ext: UnstructuredReader() for ext in unstructured_supported_exts}

        files_to_process_with_unstructured = []
        for non_csv_path in non_csv_file_paths:
            file_hash = calculate_file_hash(non_csv_path)
            if file_hash in processed_hashes:
                print(f"Skipping already processed file: {non_csv_path}")
                continue
            files_to_process_with_unstructured.append(non_csv_path)
            newly_processed_files.append({'filepath': non_csv_path, 'hash': file_hash})

        if not all_documents and not files_to_process_with_unstructured:
            print("No new documents to add.")
            return

        if files_to_process_with_unstructured:
            reader = SimpleDirectoryReader(
                input_files=files_to_process_with_unstructured,
                file_extractor=file_extractor,
                recursive=False
            )
            non_csv_documents = reader.load_data()
            all_documents.extend(non_csv_documents)

        # Chunk documents into nodes
        node_parser = service_context.node_parser
        nodes = node_parser.get_nodes_from_documents(all_documents)

        extracted_entities_list = []
        extracted_relationships_list = []
        entity_documents = [] # For entity vector store

        print("Extracting entities and relationships from document chunks...")
        for i, node in enumerate(nodes):
            try:
                prompt = extraction_prompt_template.format(text=node.text)
                response = service_context.llm.complete(prompt)
                json_output = response.text
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
                traceback.print_exc() # Print full traceback for debugging

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
                    temp_response = service_context.llm.complete(temp_prompt)
                    temp_json_output = temp_response.text
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
                            response = service_context.llm.complete(prompt)
                            json_output = response.text
                            summary_dict = parse_llm_json_output(json_output)
                            
                            if summary_dict:
                                summary_dict['community_id'] = community_id # Ensure community_id is set
                                extracted_community_summaries.append(summary_dict)
                                community_summary_documents.append(Document(text=summary_dict.get('summary', ''), extra_info=summary_dict))
                                print(f"  Summarized community {community_id} (Level {community_level})")
                        except Exception as e:
                            print(f"  Error summarizing community {community_id}: {e}")
            
                if extracted_community_summaries:
                    new_summaries_df = pd.DataFrame(extracted_community_summaries)
                    existing_summaries_df = load_community_summaries_db(output_dir)
                    updated_summaries_df = pd.concat([existing_summaries_df, new_summaries_df], ignore_index=True).drop_duplicates(subset=['community_id'])
                    save_community_summaries_db(updated_summaries_df, output_dir)
                    print(f"Saved {len(new_summaries_df)} new community summaries to {output_dir}/community_summaries.parquet")

                # Create/Update community summary vector index
                if community_summary_documents:
                    if community_vector_store:
                        community_storage_context = StorageContext.from_defaults(vector_store=community_vector_store)
                        community_index = VectorStoreIndex(community_summary_documents, storage_context=community_storage_context, service_context=service_context)
                    else:
                        community_index_dir = os.path.join(output_dir, "community_summaries_index")
                        os.makedirs(community_index_dir, exist_ok=True)
                        community_index = VectorStoreIndex(community_summary_documents, service_context=service_context)
                        community_index.storage_context.persist(persist_dir=community_index_dir)
                    print("Community summary vector index updated.")
                else:
                    print("No community summaries extracted for indexing.")
            else:
                print("No communities detected.")
        else:
            print("No relationships extracted for community detection.")

        # Create/Update main text index
        if vector_store:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context, service_context=service_context)
        else:
            index = VectorStoreIndex(nodes, service_context=service_context)
            index.storage_context.persist(persist_dir=output_dir)
        print("Main text index updated.")

        # Create/Update entity vector index
        if entity_documents:
            if entity_vector_store:
                entity_storage_context = StorageContext.from_defaults(vector_store=entity_vector_store)
                entity_index = VectorStoreIndex(entity_documents, storage_context=entity_storage_context, service_context=service_context)
            else:
                # If no specific entity_vector_store, use default storage for entities
                entity_index_dir = os.path.join(output_dir, "entities_index")
                os.makedirs(entity_index_dir, exist_ok=True)
                entity_index = VectorStoreIndex(entity_documents, service_context=service_context)
                entity_index.storage_context.persist(persist_dir=entity_index_dir)
            print("Entity vector index updated.")
        else:
            print("No entities extracted for indexing.")

        # Update and save the processed files database
        updated_df = pd.concat([processed_files_df, pd.DataFrame(newly_processed_files)], ignore_index=True)
        save_processed_files_db(updated_df, output_dir)

        print("Documents and entities processed successfully.")
    except Exception as e:
        print(f"Error adding documents: {e}")
        traceback.print_exc() # Print full traceback for debugging
