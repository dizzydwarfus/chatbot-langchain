from langchain.document_loaders import TextLoader
import os


def test_text_loader(file_path, encoding="utf-8"):
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return

        print(f"File found at: {file_path}")

        # Check file permissions
        if not os.access(file_path, os.R_OK):
            print(f"File is not readable: {file_path}")
            return

        print(f"File is readable: {file_path}")

        # Create a TextLoader instance with the given file path and encoding
        loader = TextLoader(file_path, encoding=encoding)

        # Attempt to load the document
        documents = loader.load()

        print("File loaded successfully!")
        print("Number of documents loaded:", len(documents))

        # Print the first document's content
        if documents:
            print(
                "First document content:", documents[0].page_content[500]
            )  # Print first 500 characters

        return documents

    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
        print("Try using a different encoding, such as 'latin-1'.")
    except Exception as e:
        print(f"Error loading file: {e}")
        raise


# Test with the absolute path to your sample text file and utf-8 encoding
file_path = os.path.abspath("../data/sample_text.txt")
test_text_loader(file_path, encoding="utf-8")
