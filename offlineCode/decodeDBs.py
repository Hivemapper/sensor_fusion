import os
import base64
import subprocess
import argparse
import sqlite3


def is_base64_encoded(file_path):
    """Check if the file is base64 encoded."""
    try:
        with open(file_path, "rb") as f:
            snippet = f.read(64)  # Read a small portion of the file
            base64.b64decode(snippet)
        return True
    except (base64.binascii.Error, ValueError):
        return False


def is_sqlite3(filename):
    """Check if the file is a valid SQLite3 database."""
    try:
        with sqlite3.connect(filename) as conn:
            conn.execute("PRAGMA integrity_check;")
        return True
    except sqlite3.DatabaseError:
        return False


def decode_and_repair_files(input_dir):
    # Ensure output directory exists
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isfile(item_path):
            output_dir = os.path.join(input_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, item)

            # Check if the file is base64 encoded
            if is_base64_encoded(item_path):
                # Decode the file and save to output directory
                with open(item_path, "rb") as encoded_file:
                    decoded_data = base64.b64decode(encoded_file.read())
                    with open(output_file, "wb") as decoded_file:
                        decoded_file.write(decoded_data)
                print(f"Decoded: {output_file}")

                # Check if the decoded file is a valid SQLite database
                if is_sqlite3(output_file):
                    print(
                        f"Decoded file {output_file} is a valid SQLite database. Writing to output directory."
                    )
                else:
                    print(
                        f"Decoded file {output_file} is malformed. Attempting to repair."
                    )
                    recovered_file = os.path.join(output_dir, f"recovered_{item}")
                    sql_file = f"{os.path.splitext(output_file)[0]}.sql"

                    # Repair the SQLite database
                    subprocess.run(
                        ["sqlite3", output_file, ".recover"], stdout=open(sql_file, "w")
                    )
                    subprocess.run(
                        ["sqlite3", recovered_file], stdin=open(sql_file, "r")
                    )
                    os.remove(sql_file)

                    print(f"Recovered: {recovered_file}")
            else:
                # If the file is not base64 encoded, check if it is a valid SQLite database
                print(
                    f"File {item_path} is not base64 encoded. Checking for SQLite integrity."
                )
                subprocess.run(["cp", item_path, output_file])

                if is_sqlite3(output_file):
                    print(
                        f"File {output_file} is a valid SQLite database. Writing to output directory."
                    )
                else:
                    print(f"File {output_file} is malformed. Attempting to repair.")
                    recovered_file = os.path.join(output_dir, f"recovered_{item}")
                    sql_file = f"{os.path.splitext(output_file)[0]}.sql"

                    # Repair the SQLite database
                    subprocess.run(
                        ["sqlite3", output_file, ".recover"], stdout=open(sql_file, "w")
                    )
                    subprocess.run(
                        ["sqlite3", recovered_file], stdin=open(sql_file, "r")
                    )
                    os.remove(sql_file)

                    print(f"Recovered: {recovered_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Decode and repair base64 encoded SQLite files."
    )
    parser.add_argument("top_level_dir", type=str, help="Top level input directory")
    args = parser.parse_args()

    top_level_dir = args.top_level_dir

    # Loop through each subdirectory in the top level directory
    for sub_dir in os.listdir(top_level_dir):
        sub_dir_path = os.path.join(top_level_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            # Call function to decode and repair files for each subdirectory
            decode_and_repair_files(sub_dir_path)

    print("Decoding and recovery completed for all subdirectories.")


if __name__ == "__main__":
    main()
