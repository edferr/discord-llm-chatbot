# generate_requirements.py

import subprocess

# Run pip freeze command and capture output
result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)

# Split output into lines and filter out comments and versions
packages = []
for line in result.stdout.splitlines():
    if not line.startswith('#') and '==' in line:
        package_name = line.split('==')[0]
        packages.append(package_name)

# Write packages to requirements.txt file
with open('requirements.txt', 'w') as f:
    for package in packages:
        f.write(f"{package}\n")

print(f"Successfully generated requirements.txt with {len(packages)} packages.")
