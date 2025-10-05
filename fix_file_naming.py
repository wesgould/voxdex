#!/usr/bin/env python3
"""
Utility script to fix file naming conventions for existing transcripts.
Adds proper prefixes (IM_, TWIT_, SN_) to files based on their podcast directory.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def get_prefix_mapping() -> Dict[str, str]:
    """Return mapping of podcast directory patterns to file prefixes"""
    return {
        "Intelligent_Machines": "IM_",
        "This_Week_in_Tech": "TWIT_",
        "Security_Now": "SN_"
    }

def find_files_to_rename(output_dir: Path) -> List[Tuple[Path, str]]:
    """Find all files that need renaming and their required prefix"""
    files_to_rename = []
    prefix_mapping = get_prefix_mapping()

    # File extensions to process
    extensions = {".txt", ".json", ".srt"}

    for podcast_dir in output_dir.iterdir():
        if not podcast_dir.is_dir():
            continue

        # Determine prefix based on podcast directory name
        prefix = None
        for pattern, prefix_value in prefix_mapping.items():
            if pattern in podcast_dir.name:
                prefix = prefix_value
                break

        if not prefix:
            print(f"Warning: No prefix mapping found for directory {podcast_dir.name}")
            continue

        # Look for episode directories
        for episode_dir in podcast_dir.iterdir():
            if not episode_dir.is_dir():
                continue

            # Find files that need renaming
            for file_path in episode_dir.iterdir():
                if file_path.suffix in extensions:
                    # Check if file already has correct prefix
                    if not file_path.stem.startswith(prefix.rstrip('_')):
                        files_to_rename.append((file_path, prefix))

    return files_to_rename

def extract_episode_number(filename: str) -> str:
    """Extract episode number from filename"""
    # Look for number at start of filename
    match = re.match(r'^(\d+)', filename)
    if match:
        return match.group(1)
    return filename.split('_')[0]  # fallback

def rename_file(file_path: Path, prefix: str, dry_run: bool = True) -> bool:
    """Rename a single file with the proper prefix"""
    original_name = file_path.name
    original_stem = file_path.stem
    extension = file_path.suffix

    # Extract episode number and type
    episode_number = extract_episode_number(original_stem)

    # Determine file type (raw, diarized, enhanced, metadata)
    if "_raw" in original_stem:
        file_type = "_raw"
    elif "_diarized" in original_stem:
        file_type = "_diarized"
    elif "_enhanced" in original_stem:
        file_type = "_enhanced"
    elif "_metadata" in original_stem:
        file_type = "_metadata"
    else:
        # Handle files that are just numbers
        if original_stem.isdigit():
            file_type = "_raw"  # assume raw if no type specified
        else:
            print(f"Warning: Could not determine file type for {original_name}")
            return False

    # Create new filename
    new_name = f"{prefix}{episode_number}{file_type}{extension}"
    new_path = file_path.parent / new_name

    print(f"{'[DRY RUN] ' if dry_run else ''}Rename: {original_name} -> {new_name}")

    if not dry_run:
        if new_path.exists():
            print(f"Error: Target file {new_name} already exists!")
            return False
        try:
            file_path.rename(new_path)
            return True
        except Exception as e:
            print(f"Error renaming {original_name}: {e}")
            return False

    return True

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Fix file naming conventions for existing transcripts")
    parser.add_argument("--output-dir", default="output", help="Output directory path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without doing it")
    parser.add_argument("--execute", action="store_true", help="Actually perform the renames")

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("Use --dry-run to preview changes or --execute to perform renames")
        return

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        return

    print(f"Scanning {output_dir} for files to rename...")
    files_to_rename = find_files_to_rename(output_dir)

    if not files_to_rename:
        print("No files found that need renaming!")
        return

    print(f"Found {len(files_to_rename)} files to rename:")
    print()

    # Group by directory for better output
    by_directory = {}
    for file_path, prefix in files_to_rename:
        episode_dir = file_path.parent
        if episode_dir not in by_directory:
            by_directory[episode_dir] = []
        by_directory[episode_dir].append((file_path, prefix))

    success_count = 0
    total_count = len(files_to_rename)

    for episode_dir, files in by_directory.items():
        print(f"\nDirectory: {episode_dir}")
        for file_path, prefix in files:
            if rename_file(file_path, prefix, dry_run=args.dry_run):
                success_count += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary: {success_count}/{total_count} files {'would be ' if args.dry_run else ''}renamed successfully")

    if args.dry_run:
        print("\nTo actually perform these renames, run with --execute")

if __name__ == "__main__":
    main()