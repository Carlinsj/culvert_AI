"""Read only permitted inventory fields with the bundled spreadsheet runtime."""
import argparse
import hashlib
import json
from pathlib import Path

import openpyxl

FIELDS = ("Small Culvert ID", "Legacy Culvert ID", "Route", "Reference Marker",
          "Ref Marker Offset (Ft)", "County", "x", "y")


def extract(path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook["Small Culverts - R8"]
    rows = iter(sheet.values)
    header = next(rows)
    positions = {field: header.index(field) for field in FIELDS}
    records = []
    total = 0
    for source_row, values in enumerate(rows, start=2):
        if not any(value is not None for value in values):
            continue
        total += 1
        if str(values[positions["County"]]).strip().upper() != "ULSTER":
            continue
        record = {field: values[position] for field, position in positions.items()}
        record["source_excel_row"] = source_row
        record["record_key"] = f"source_row_{source_row}"
        records.append(record)
    workbook.close()
    return dict(workbook_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                workbook_path=str(Path(path).resolve()), worksheet="Small Culverts - R8",
                workbook_records=total, workbook_columns=len(header), fields_read=list(FIELDS),
                records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(extract(args.workbook), indent=2, default=str) + "\n")
