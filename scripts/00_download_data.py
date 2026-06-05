from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "satcat.csv": "https://celestrak.org/pub/satcat.csv",
    "ucs_satellite_database_2023.xlsx": "https://www.ucs.org/sites/default/files/2024-01/UCS-Satellite-Database%205-1-2023.xlsx",
    "owid_objects_launched.csv": "https://ourworldindata.org/grapher/yearly-number-of-objects-launched-into-outer-space.csv?csvType=full&useColumnShortNames=true&v=1",
}

def download_file(url: str, out_path: Path) -> None:
    print(f"Descargando {out_path.name}...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    print(f"Guardado en {out_path}")

def main() -> None:
    for filename, url in FILES.items():
        download_file(url, RAW_DIR / filename)
    print("\nDescarga completada.")

if __name__ == "__main__":
    main()