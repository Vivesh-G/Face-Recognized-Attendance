"""
attendance_manager.py  –  CSV-based attendance log
"""
import os
import csv
from datetime import datetime, date

ATTENDANCE_DIR = "attendance"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

FIELDS = ["Name", "Class", "Department", "Status", "Time", "Confidence"]


def _today_path() -> str:
    return os.path.join(ATTENDANCE_DIR, f"{date.today()}.csv")


def get_csv_path() -> str:
    """Get today's attendance CSV path."""
    return _today_path()


def ensure_today(people_meta: dict) -> str:
    """Create or update today's CSV with all registered people."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    path = _today_path()

    if not os.path.exists(path):
        try:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                for name, meta in sorted(people_meta.items()):
                    writer.writerow({
                        "Name": name,
                        "Class": meta.get("class", ""),
                        "Department": meta.get("dept", ""),
                        "Status": "Absent",
                        "Time": "",
                        "Confidence": "",
                    })
            print(f"[INFO] Created attendance CSV with {len(people_meta)} people")
        except Exception as e:
            print(f"[ERROR] Failed to create attendance CSV: {e}")
    else:
        # If file exists, check if any new people need to be added
        rows = _read(path)
        existing_names = {row["Name"] for row in rows}
        new_people = []
        for name, meta in people_meta.items():
            if name not in existing_names:
                new_people.append({
                    "Name": name,
                    "Class": meta.get("class", ""),
                    "Department": meta.get("dept", ""),
                    "Status": "Absent",
                    "Time": "",
                    "Confidence": "",
                })

        if new_people:
            try:
                with open(path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    writer.writerows(new_people)
                print(f"[INFO] Added {len(new_people)} new people to today's CSV")
            except Exception as e:
                print(f"[ERROR] Failed to append to attendance CSV: {e}")

    return path



def _read(path: str) -> list[dict]:
    """Read CSV file."""
    try:
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return []


def _write(path: str, rows: list[dict]) -> None:
    """Write rows to CSV file."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"[ERROR] Failed to write {path}: {e}")


def mark_present(name: str, confidence: float, people_meta: dict) -> bool:
    """Mark a person Present only if currently Absent. Returns True if updated."""
    path = ensure_today(people_meta)
    rows = _read(path)
    for row in rows:
        if row["Name"] == name and row["Status"] == "Absent":
            row["Status"] = "Present"
            row["Time"] = datetime.now().strftime("%H:%M:%S")
            row["Confidence"] = f"{confidence*100:.1f}%"
            _write(path, rows)
            return True
    return False


def update_attendance(name: str, status: str, people_meta: dict) -> bool:
    """Update attendance status for a person."""
    path = ensure_today(people_meta)
    rows = _read(path)
    for row in rows:
        if row["Name"] == name:
            row["Status"] = status
            if status == "Present":
                row["Time"] = datetime.now().strftime("%H:%M:%S")
            _write(path, rows)
            return True
    return False


def get_attendance() -> list[dict]:
    """Get today's attendance records."""
    path = _today_path()
    if not os.path.exists(path):
        return []
    return _read(path)
