import requests
import json
from pathlib import Path

# Base URL - adjust this based on your server
BASE_URL = "http://localhost:8000/api"


def test_file_upload():
    """Test single file upload"""
    print("Testing single file upload...")
    
    # Create a test file
    test_file_path = "test_document.txt"
    with open(test_file_path, 'w') as f:
        f.write("This is a test file for the upload API")
    
    # Upload the file
    url = f"{BASE_URL}/files/upload/"
    
    with open(test_file_path, 'rb') as f:
        files = {'file': f}
        data = {'description': 'Test file upload'}
        response = requests.post(url, files=files, data=data)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_list_files():
    """Test listing all files"""
    print("\nTesting file listing...")
    
    url = f"{BASE_URL}/files/"
    response = requests.get(url)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_file_search():
    """Test file search functionality"""
    print("\nTesting file search...")
    
    url = f"{BASE_URL}/files/"
    params = {'search': 'test'}
    response = requests.get(url, params=params)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_bulk_upload():
    """Test bulk file upload"""
    print("\nTesting bulk file upload...")
    
    # Create multiple test files
    test_files = []
    for i in range(3):
        filename = f"test_file_{i}.txt"
        with open(filename, 'w') as f:
            f.write(f"Content of test file {i}")
        test_files.append(filename)
    
    # Upload multiple files
    url = f"{BASE_URL}/files/bulk-upload/"
    
    files = []
    for filename in test_files:
        files.append(('files', open(filename, 'rb')))
    
    data = {'description': 'Bulk upload test'}
    response = requests.post(url, files=files, data=data)
    
    # Clean up file handles
    for _, file_handle in files:
        file_handle.close()
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_file_stats():
    """Test file statistics endpoint"""
    print("\nTesting file statistics...")
    
    url = f"{BASE_URL}/files/stats/"
    response = requests.get(url)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


# CURL Examples
print("""
================================
CURL COMMAND EXAMPLES
================================

1. Upload a single file:
curl -X POST http://localhost:8000/api/files/upload/ \\
  -F "file=@/path/to/your/file.pdf" \\
  -F "description=Important document"

2. List all files:
curl http://localhost:8000/api/files/

3. List files with search:
curl "http://localhost:8000/api/files/?search=document"

4. List files with filtering by type:
curl "http://localhost:8000/api/files/?file_type=.pdf"

5. List files with custom ordering:
curl "http://localhost:8000/api/files/?ordering=-file_size"

6. Get specific file details:
curl http://localhost:8000/api/files/1/

7. Delete a file:
curl -X DELETE http://localhost:8000/api/files/1/

8. Upload multiple files:
curl -X POST http://localhost:8000/api/files/bulk-upload/ \\
  -F "files=@file1.pdf" \\
  -F "files=@file2.jpg" \\
  -F "files=@file3.docx" \\
  -F "description=Multiple files upload"

9. Get file statistics:
curl http://localhost:8000/api/files/stats/

================================
""")

if __name__ == "__main__":
    print("Running API tests...")
    print("Make sure your Django server is running on localhost:8000")
    print("=" * 50)
    
    try:
        # Run tests
        test_file_upload()
        test_list_files()
        test_file_search()
        test_bulk_upload()
        test_file_stats()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print("Make sure Django server is running: python manage.py runserver")
    except Exception as e:
        print(f"Error: {e}")