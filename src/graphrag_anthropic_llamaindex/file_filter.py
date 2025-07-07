import fnmatch
import os
from typing import List, Optional
from pathlib import Path
from llama_index.core.schema import Document


class FileFilter:
    """
    Handles file filtering based on ignore patterns.
    
    This class encapsulates the logic for determining which files should be ignored
    during document processing based on configurable glob patterns.
    """
    
    def __init__(self, ignore_patterns: Optional[List[str]] = None):
        """
        Initialize FileFilter with ignore patterns.
        
        Args:
            ignore_patterns: List of glob patterns to ignore. If None, no files are ignored.
        """
        self.ignore_patterns = ignore_patterns or []
    
    def should_ignore(self, file_path: str) -> bool:
        """
        Check if a file should be ignored based on ignore patterns.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if file should be ignored, False otherwise
        """
        if not self.ignore_patterns:
            return False
        
        # Normalize path separators for cross-platform compatibility
        normalized_path = file_path.replace('\\', '/')
        filename = os.path.basename(file_path)
        
        # Check against all patterns
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(normalized_path, pattern) or fnmatch.fnmatch(filename, pattern):
                return True
        
        return False
    
    def filter_file_paths(self, file_paths: List[str]) -> List[str]:
        """
        Filter a list of file paths, removing ignored files.
        
        Args:
            file_paths: List of file paths to filter
            
        Returns:
            List[str]: Filtered list of file paths
        """
        filtered_paths = []
        for file_path in file_paths:
            if self.should_ignore(file_path):
                print(f"Ignoring file: {file_path}")
            else:
                filtered_paths.append(file_path)
        return filtered_paths
    
    def filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        Filter a list of documents, removing those from ignored files.
        
        Args:
            documents: List of Document objects to filter
            
        Returns:
            List[Document]: Filtered list of documents
        """
        filtered_docs = []
        for doc in documents:
            # Get file path from document metadata
            file_path = doc.extra_info.get('file_name', '') or doc.extra_info.get('virtual_path', '')
            
            if self.should_ignore(file_path):
                print(f"Ignoring document: {file_path}")
            else:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def find_files(self, input_dir: str, extensions: Optional[List[str]] = None, recursive: bool = True) -> List[str]:
        """
        Find files in directory with optional extension filtering and ignore patterns.
        
        Args:
            input_dir: Directory to search
            extensions: List of file extensions to include (e.g., ['.txt', '.pdf'])
            recursive: Whether to search recursively
            
        Returns:
            List[str]: List of file paths that match criteria and are not ignored
        """
        file_paths = []
        
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check extension if specified
                    if extensions and Path(file).suffix.lower() not in extensions:
                        continue
                    
                    # Check ignore patterns
                    if not self.should_ignore(file_path):
                        file_paths.append(file_path)
                    else:
                        print(f"Ignoring file: {file_path}")
        else:
            for file in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file)
                
                if not os.path.isfile(file_path):
                    continue
                
                # Check extension if specified
                if extensions and Path(file).suffix.lower() not in extensions:
                    continue
                
                # Check ignore patterns
                if not self.should_ignore(file_path):
                    file_paths.append(file_path)
                else:
                    print(f"Ignoring file: {file_path}")
        
        return file_paths