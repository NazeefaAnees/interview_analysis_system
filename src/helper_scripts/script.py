import os

# Create files for "se_jd_XX.txt"
for i in range(1, 21):
    filename = f"se_jd_{i:02d}.txt"
    with open(filename, 'w') as f:
        pass  # This will create an empty file

# Create files for "ds_jd_XX.txt"
for i in range(1, 21):
    filename = f"ds_jd_{i:02d}.txt"
    with open(filename, 'w') as f:
        pass  # This will create an empty file
