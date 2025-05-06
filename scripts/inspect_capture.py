#!/usr/bin/env python
"""
Inspection utility for Positronic Brain capture files.

This script loads a .pt file from the captures directory and
creates a human-readable representation of its contents in a text file.
"""

import torch
import sys
import os
import argparse
from pprint import pprint
import datetime

# Add project root to Python path to allow importing positronic_brain modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local implementation of safe_load
def safe_load(filename, default=None, device=None):
    """Load an object from a file with error handling.
    
    Args:
        filename: Path to the file to load
        default: Default value to return if file doesn't exist
        device: Device to map tensors to (if applicable)
        
    Returns:
        The loaded object or default if file doesn't exist
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}, returning default", file=sys.stderr)
        return default
        
    try:
        if device is not None:
            return torch.load(filename, map_location=device)
        else:
            return torch.load(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}", file=sys.stderr)
        if default is not None:
            print(f"Returning default value instead", file=sys.stderr)
            return default
        raise

def print_tensor_summary(tensor, max_elems=20, full_output=False):
    """Prints a summary of a tensor's shape, dtype, device, and optionally some data."""
    if not isinstance(tensor, torch.Tensor):
        # If it's not a tensor, use standard repr
        try:
            representation = repr(tensor)
            if len(representation) > 150: # Limit long representations
                 return f"{representation[:75]}...{representation[-75:]} (type: {type(tensor).__name__})"
            return f"{representation} (type: {type(tensor).__name__})"
        except Exception as e:
            return f"<Error representing object: {e}> (type: {type(tensor).__name__})"

    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    try:
        device = tensor.device
    except AttributeError: # Handle tensors that might not have .device (e.g., loaded on CPU?)
        device = 'cpu?'
    numel = tensor.numel()

    summary = f"Tensor(shape={shape}, dtype={dtype}, device='{device}', numel={numel}"

    # Only show data preview if the tensor isn't huge
    if numel == 0:
        summary += ")"
    elif full_output:
        # Print full data if requested, with a warning for very large tensors
        if numel > 10000:
            summary += ", data=<Tensor too large for full display>)"
            print(f"WARNING: Tensor has {numel} elements. Full display skipped for brevity even with --full flag.", file=sys.stderr)
        else:
            try:
                summary += f", data={tensor.tolist()})"
            except Exception as e:
                summary += f", data=<Error converting to list: {e}>)"
    elif numel <= max_elems * 2 and numel < 100: # Show full data for small tensors
        summary += f", data={tensor.tolist()})"
    elif numel < 2000: # Show partial data for moderately sized tensors
        try:
            first_elems = tensor.flatten()[:max_elems].tolist()
            last_elems = tensor.flatten()[-max_elems:].tolist()
            # Format numbers nicely
            first_str = ', '.join([f"{x:.2f}" if isinstance(x, float) else str(x) for x in first_elems])
            last_str = ', '.join([f"{x:.2f}" if isinstance(x, float) else str(x) for x in last_elems])
            summary += f", data=[{first_str}...{last_str}])"
        except Exception: # Fallback if flattening/listing fails
             summary += ", data=<preview unavailable>)"
    else: # Don't show data for very large tensors
        summary += ", data=<large tensor>)"
    return summary

def inspect_data(data, file_obj, indent=0, max_depth=10, full_output=False):
    """Recursively writes the contents of loaded data to a file with controlled depth."""
    prefix = "  " * indent
    if indent >= max_depth:
        print(f"{prefix}<Max recursion depth reached>", file=file_obj)
        return

    if isinstance(data, dict):
        print(f"{prefix}{{\n", file=file_obj)
        for key, value in data.items():
            print(f"{prefix}  '{key}': ", end="", file=file_obj)
            # Check if value is a tensor before recursing
            if isinstance(value, torch.Tensor):
                print(print_tensor_summary(value, full_output=full_output), file=file_obj)
            else:
                inspect_data(value, file_obj, indent + 2, max_depth, full_output)
        print(f"{prefix}}}", file=file_obj)
    elif isinstance(data, list):
        list_len = len(data)
        print(f"{prefix}[", file=file_obj)

        if full_output:
            if list_len > 1000:
                print(f"{prefix}  ... (List too long: {list_len} items. Displaying first/last 20 even with --full) ...", file=file_obj)
                items_to_show_indices = list(range(20)) + list(range(list_len - 20, list_len))
                ellipsis_printed = False
                for i, item in enumerate(data):
                    if i in items_to_show_indices:
                        print(f"{prefix}  - ", end="", file=file_obj)
                        if isinstance(item, torch.Tensor):
                            print(print_tensor_summary(item, full_output=full_output), file=file_obj)
                        else:
                            inspect_data(item, file_obj, indent + 2, max_depth, full_output)
                    elif not ellipsis_printed and i == 20:
                        print(f"{prefix}    ...", file=file_obj)
                        ellipsis_printed = True
            else:
                # Print all items in full mode for reasonable lists
                for i, item in enumerate(data):
                    print(f"{prefix}  - [{i}] ", end="", file=file_obj)
                    if isinstance(item, torch.Tensor):
                        print(print_tensor_summary(item, full_output=full_output), file=file_obj)
                    else:
                        inspect_data(item, file_obj, indent + 2, max_depth, full_output)
        else:
            # Summarized list printing
            max_list_items_preview = 10  # Increased preview limit
            items_to_show_indices = list(range(min(max_list_items_preview, list_len)))
            if list_len > max_list_items_preview * 2:
                items_to_show_indices.extend(list(range(list_len - max_list_items_preview, list_len)))

            ellipsis_printed = False
            for i, item in enumerate(data):
                if i in items_to_show_indices:
                    print(f"{prefix}  - [{i}] ", end="", file=file_obj)
                    if isinstance(item, torch.Tensor):
                        print(print_tensor_summary(item, full_output=full_output), file=file_obj)
                    else:
                        inspect_data(item, file_obj, indent + 2, max_depth, full_output)
                elif not ellipsis_printed and i == max_list_items_preview and list_len > max_list_items_preview * 2:
                    print(f"{prefix}  ... ({list_len - max_list_items_preview * 2} more items) ...", file=file_obj)
                    ellipsis_printed = True

        print(f"{prefix}]", file=file_obj)

    elif isinstance(data, torch.Tensor):
        # This case handles tensors directly within a list/dict if not caught above
        print(f"{prefix}{print_tensor_summary(data, full_output=full_output)}", file=file_obj)
    else:
        # Use standard repr for other simple types
        try:
            representation = repr(data)
            if len(representation) > 150 and not full_output: # Limit long representations if not full output
                 print(f"{prefix}{representation[:75]}...{representation[-75:]} (type: {type(data).__name__})", file=file_obj)
            else:
                 print(f"{prefix}{representation} (type: {type(data).__name__})", file=file_obj)
        except Exception as e:
             print(f"{prefix}<Error representing object: {e}> (type: {type(data).__name__})", file=file_obj)


def main():
    parser = argparse.ArgumentParser(description="Inspect the contents of a serialized .pt capture file and write to a text file.")
    parser.add_argument("filepath", help="Path to the .pt file to inspect (e.g., tests/captures/step1_output_short_fox.pt).")
    parser.add_argument("--output", "-o", help="Output file path (default: auto-generate based on input filename)")
    parser.add_argument("--format", "-f", choices=["txt", "md"], default="md", help="Output format (default: md)")
    parser.add_argument("--full", action="store_true", help="Show full contents of lists and tensors, disable summarization.")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found at '{args.filepath}'", file=sys.stderr)
        print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
        print("Please provide a valid relative or absolute path.", file=sys.stderr)
        sys.exit(1)

    # Create output filename if not specified
    if args.output:
        output_file = args.output
    else:
        basename = os.path.basename(args.filepath).replace('.pt', '')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(project_root, "inspection_reports")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{basename}_{timestamp}.{args.format}")

    try:
        # Load the data using the safe_load utility
        loaded_data = safe_load(args.filepath)
        
        # Open the output file and write the inspection
        with open(output_file, 'w') as f:
            if args.format == "md":
                # Write markdown header
                f.write(f"# Inspection Report: {os.path.basename(args.filepath)}\n\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"File path: `{os.path.abspath(args.filepath)}`\n\n")
                f.write("## File Contents\n\n")
                f.write("```\n")  # Start code block for better formatting
            else:
                # Write plain text header
                f.write(f"=== Inspection Report: {os.path.basename(args.filepath)} ===\n\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"File path: {os.path.abspath(args.filepath)}\n\n")
                f.write("=== File Contents ===\n\n")
            
            # Inspect and write the loaded data structure
            inspect_data(loaded_data, f, full_output=args.full)
            
            if args.format == "md":
                f.write("```\n")  # End code block
        
        print(f"Inspection report written to: {output_file}")
        
        # Also print the first few lines to confirm successful output
        with open(output_file, 'r') as f:
            preview = "\n".join(f.readlines()[:10]) + "\n..."
            print("\nPreview of the report:")
            print("-" * 40)
            print(preview)
            print("-" * 40)

    except FileNotFoundError:
        # This shouldn't happen due to the check above, but belt-and-suspenders
        print(f"Error: File not found during loading '{args.filepath}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading or inspecting file: {type(e).__name__} - {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
