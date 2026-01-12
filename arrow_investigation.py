import pandas as pd

# Load the data
df = pd.read_csv('./DATASETS_excalidraw/file_lifecycle_events.csv', encoding='utf-8')

# Filter rows with => arrows
arrow_files = df[df['filepath'].str.contains('=>', na=False)]

print("=" * 80)
print("DEEPER INVESTIGATION: Files with '=>' marked as 'modified'")
print("=" * 80)
print()

print(f"Total files with '=>' notation: {len(arrow_files)}")
print(f"All marked as event_type: {arrow_files['event_type'].unique()}")
print()

# Look at a specific example in detail
print("Detailed example #1:")
print("-" * 80)
sample = arrow_files.iloc[0]
print(f"Filepath: {sample['filepath']}")
print(f"Event type: {sample['event_type']}")
print(f"Commit hash: {sample['commit_hash']}")
print(f"Commit subject: {sample['commit_subject']}")
print(f"Old path: {sample['old_path']}")
print(f"New path: {sample['new_path']}")
print(f"Additions: {sample['additions']}")
print(f"Deletions: {sample['deletions']}")
print()

# Check if there are actual renamed events in the dataset
print("=" * 80)
print("Checking for properly recorded 'renamed' events:")
print("=" * 80)
renamed = df[df['event_type'] == 'renamed']
print(f"Total 'renamed' events: {len(renamed)}")
print()

if len(renamed) > 0:
    print("Sample renamed events:")
    for idx, row in renamed.head(3).iterrows():
        print(f"\nFilepath: {row['filepath']}")
        print(f"Old path: {row['old_path']}")
        print(f"New path: {row['new_path']}")
else:
    print("No events with event_type='renamed' found!")
    print()

# Let's look at the pattern more carefully
print("=" * 80)
print("Pattern analysis of '=>' in filepaths:")
print("=" * 80)
print()

# Group by the pattern to see common types
patterns = arrow_files['filepath'].head(20)
print("First 20 filepaths with '=>':")
for i, fp in enumerate(patterns, 1):
    print(f"{i:2d}. {fp}")

print()
print("=" * 80)
print("HYPOTHESIS: The Issue")
print("=" * 80)
print()
print("The script appears to have a BUG in parsing git's --raw output.")
print()
print("What's happening:")
print("1. Git's --numstat output uses '=>' notation for renamed files")
print("2. The script is reading this notation into the 'filepath' column")
print("3. BUT it's marking these as 'modified' instead of 'renamed'")
print("4. The old_path and new_path columns are empty (NaN)")
print()
print("The bug is likely in how the script matches --raw output with --numstat output.")
print("The --numstat lines contain the '=>' notation, but the script isn't properly")
print("matching them with the corresponding --raw parse results that contain the")
print("actual old_path and new_path information.")
print()

# Let's verify this hypothesis by checking the commit
print("=" * 80)
print("Verification: Check what Git actually shows for one of these commits")
print("=" * 80)
print()
sample_commit = arrow_files.iloc[0]['commit_hash']
sample_filepath = arrow_files.iloc[0]['filepath']
print(f"Commit: {sample_commit}")
print(f"Filepath with '=>': {sample_filepath}")
print()
print("To verify, run this git command in the excalidraw repo:")
print(f"  git show --raw --numstat {sample_commit} | grep '=>'")
print()
print("Expected: Git's --raw output should show 'R' (rename) status,")
print("but the script may be failing to parse it correctly.")
print()

# Check if there are patterns in commit subjects
print("=" * 80)
print("Commit subjects for files with '=>' (top 10 unique):")
print("=" * 80)
print()
for subject in arrow_files['commit_subject'].unique()[:10]:
    count = len(arrow_files[arrow_files['commit_subject'] == subject])
    print(f"[{count:3d} files] {subject}")