import os
import argparse
import nibabel as nib


def convert_tck_to_trk(tck_path, output_path):
    """Convert a single TCK file to TRK format."""

    print(f"Converting: {tck_path}")
    print(f"Output:    {output_path}")

    # Load TCK tractogram
    sft = nib.streamlines.load(tck_path)

    # Save as TRK
    nib.streamlines.save(sft.tractogram, output_path)

    print("Conversion completed.")


def convert_directory(input_dir, output_dir):
    """Convert all TCK files in a directory to TRK format."""

    os.makedirs(output_dir, exist_ok=True)

    tck_files = [
        filename
        for filename in os.listdir(input_dir)
        if filename.lower().endswith(".tck")
    ]

    if not tck_files:
        print(f"No .tck files found in: {input_dir}")
        return

    for filename in tck_files:

        tck_path = os.path.join(input_dir, filename)

        # Replace .tck with .trk
        trk_filename = os.path.splitext(filename)[0] + ".trk"
        trk_path = os.path.join(output_dir, trk_filename)

        print(f"\nConverting {filename} -> {trk_filename}")

        try:
            sft = nib.streamlines.load(tck_path)
            nib.streamlines.save(sft.tractogram, trk_path)
            print("Done.")

        except Exception as e:
            print(f"ERROR converting {filename}: {e}")

    print("\nAll conversions completed.")


def main():

    parser = argparse.ArgumentParser(
        description="Convert TCK tractogram files to TRK format."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .tck file or directory containing .tck files."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output .trk file or output directory."
    )

    args = parser.parse_args()

    # Single TCK file
    if os.path.isfile(args.input):

        if not args.input.lower().endswith(".tck"):
            raise ValueError("Input file must have a .tck extension.")

        convert_tck_to_trk(
            args.input,
            args.output
        )

    # Directory containing TCK files
    elif os.path.isdir(args.input):

        convert_directory(
            args.input,
            args.output
        )

    else:
        raise FileNotFoundError(
            f"Input path does not exist: {args.input}"
        )


if __name__ == "__main__":
    main()
