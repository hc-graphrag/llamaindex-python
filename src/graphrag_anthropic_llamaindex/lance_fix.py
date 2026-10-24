"""
LanceDB compatibility fixes for empty query builder nprobes issue.
"""

import logging
from typing import Any, List
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)


class FixedLanceDBVectorStore(LanceDBVectorStore):
    """
    LanceDBVectorStore with fixes for LanceEmptyQueryBuilder nprobes error.

    This wrapper class fixes the issue where LanceDB returns LanceEmptyQueryBuilder
    when tables have no vector index, causing AttributeError on nprobes.
    """

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query the vector store with error handling for empty query builder.

        Args:
            query: The vector store query
            **kwargs: Additional keyword arguments

        Returns:
            VectorStoreQueryResult with proper error handling
        """
        try:
            # Try the original query first
            return super().query(query, **kwargs)

        except AttributeError as e:
            if "'LanceEmptyQueryBuilder' object has no attribute 'nprobes'" not in str(e):
                raise
            logger.warning(f"LanceDB の未インデックステーブルで nprobes エラー。直接クエリを実行します。Table: {self._table_name}")
            return self._direct_lancedb_query(query, **kwargs)

    def _direct_lancedb_query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Perform direct LanceDB query bypassing llama-index wrapper issues.

        Args:
            query: The vector store query
            **kwargs: Additional keyword arguments

        Returns:
            VectorStoreQueryResult from direct LanceDB query
        """
        if self._table is None:
            raise RuntimeError("Table is not initialized")

        # Perform direct LanceDB search
        search_builder = self._table.search(
            query.query_embedding,
            vector_column_name=self.vector_column_name
        )

        # Apply limit
        search_builder = search_builder.limit(query.similarity_top_k * self.overfetch_factor)

        # Apply filters if any
        if query.filters is not None:
            where = self._to_lance_filter_direct(query.filters)
            if where:
                search_builder = search_builder.where(where)

        # Execute query
        results = search_builder.to_pandas()

        if len(results) == 0:
            logger.info(f"Direct LanceDB query returned 0 results for table {self._table_name}")
            return VectorStoreQueryResult(
                nodes=[],
                similarities=[],
                ids=[]
            )

        # Convert results to nodes
        nodes = []
        for _, item in results.iterrows():
            try:
                from llama_index.core.vector_stores.utils import metadata_dict_to_node
                node = metadata_dict_to_node(item.metadata)
                node.embedding = list(item[self.vector_column_name])
            except Exception:
                # Fallback to basic node creation
                from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
                metadata = item.metadata if hasattr(item, 'metadata') and item.metadata else {}
                node = TextNode(
                    text=item[self.text_key] or "",
                    id_=item.id,
                    metadata=metadata,
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(
                            node_id=item.get(self.doc_id_key, item.id)
                        ),
                    },
                )
                if self.vector_column_name in results.columns:
                    node.embedding = list(item[self.vector_column_name])

            nodes.append(node)

        # Calculate similarities
        from llama_index.vector_stores.lancedb.base import _to_llama_similarities
        similarities = _to_llama_similarities(results)

        logger.info(f"Direct LanceDB query returned {len(nodes)} results for table {self._table_name}")

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=results["id"].tolist(),
        )

    def _to_lance_filter_direct(self, filters) -> str:
        """
        Convert llama-index filters to Lance filter string.
        Simplified version for direct queries.
        """
        # For now, return empty string (no filters)
        # Can be extended to support filters if needed
        return ""

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes and create vector index if it doesn't exist.

        Args:
            nodes: List of nodes to add
            **add_kwargs: Additional keyword arguments

        Returns:
            List of node IDs added
        """
        # Add nodes using parent class
        result = super().add(nodes, **add_kwargs)

        # Try to create vector index if we successfully added nodes
        if result and self._table is not None:
            try:
                # Check if vector index exists by trying to query
                test_query = VectorStoreQuery(
                    query_embedding=[0.0] * 384,  # Dummy embedding for test
                    similarity_top_k=1
                )
                self._test_vector_index(test_query)

            except AttributeError as e:
                if "'LanceEmptyQueryBuilder' object has no attribute 'nprobes'" in str(e):
                    logger.info(f"Creating vector index for table {self._table_name}")
                    try:
                        self.create_index()
                        logger.info(f"Vector index created successfully for table {self._table_name}")
                    except Exception as index_error:
                        logger.warning(f"Failed to create vector index: {index_error}")

        return result

    def _test_vector_index(self, query: VectorStoreQuery) -> None:
        """
        Test if vector index is available by attempting a query.

        Args:
            query: Test query to check index availability

        Raises:
            AttributeError: If LanceEmptyQueryBuilder is returned (no index)
        """
        # This method will raise AttributeError if index doesn't exist
        super().query(query)

    def ensure_vector_index(self) -> bool:
        """
        Ensure that a vector index exists on the table.
        For small datasets that can't create indices, this method will
        attempt alternative approaches.

        Returns:
            True if index exists or was created successfully, False otherwise
        """
        if self._table is None:
            logger.warning("Table is not initialized, cannot create index")
            return False

        try:
            # Test if index exists
            test_query = VectorStoreQuery(
                query_embedding=[0.0] * 384,  # Dummy embedding
                similarity_top_k=1
            )
            self._test_vector_index(test_query)
            logger.debug(f"Vector index already exists for table {self._table_name}")
            return True

        except AttributeError as e:
            if "'LanceEmptyQueryBuilder' object has no attribute 'nprobes'" in str(e):
                logger.info(f"No vector index found for table {self._table_name}")

                # Check table size
                row_count = self._table.count_rows()
                logger.info(f"Table {self._table_name} has {row_count} rows")

                if row_count < 256:
                    logger.info(f"Small dataset ({row_count} rows), LanceDB will use brute force search")
                    # For small datasets, we can't create indices but queries should still work
                    # The issue is likely in the nprobes call, which our wrapper handles
                    return True
                else:
                    logger.info(f"Large enough dataset, attempting to create vector index...")
                    try:
                        self.create_index(
                            num_partitions=max(1, min(256, row_count // 10)),
                            num_sub_vectors=min(96, max(1, row_count // 20))
                        )
                        logger.info(f"Vector index created successfully for table {self._table_name}")
                        return True
                    except Exception as index_error:
                        logger.warning(f"Failed to create vector index, will use brute force: {index_error}")
                        return True  # Still return True as brute force should work
            else:
                logger.error(f"Unexpected error testing vector index: {e}")
                return False
        except Exception as e:
            logger.error(f"Error ensuring vector index: {e}")
            return False


def create_fixed_lancedb_store(*args, **kwargs) -> FixedLanceDBVectorStore:
    """
    Factory function to create FixedLanceDBVectorStore with the same interface as LanceDBVectorStore.

    Returns:
        FixedLanceDBVectorStore instance
    """
    return FixedLanceDBVectorStore(*args, **kwargs)