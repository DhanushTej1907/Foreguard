import zipfile
import os

zip_name = 'NeuralFabric_Final.zip'
exclude_dirs = {'.venv', '__pycache__', '.git', '.pytest_cache', '.idea', '.vscode', 'paper_renders', 'explanation_renders'}

print("Zipping project...")
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        # Modify dirs in-place to skip excluded directories entirely
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file == zip_name or file.endswith('.zip') or file.endswith(('.aux', '.log', '.out', '.pyc')):
                continue
                
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, '.')
            zf.write(file_path, arcname)

size_mb = os.path.getsize(zip_name) / (1024 * 1024)
print(f"Success! Created {zip_name}")
print(f"Final Size: {size_mb:.2f} MB")
