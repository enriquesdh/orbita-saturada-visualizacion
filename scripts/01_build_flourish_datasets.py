from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FLOURISH_DIR = ROOT_DIR / "data" / "flourish"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FLOURISH_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DATE = pd.Timestamp("2026-04-17")
MAX_COMPLETE_YEAR = 2025


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\n", " ", regex=False)
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def save_csv(df: pd.DataFrame, filename: str) -> None:
    path = FLOURISH_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Guardado {path.relative_to(ROOT_DIR)} -> {df.shape}")



def translate_country_name(name: object) -> str:
    if pd.isna(name):
        return ""

    translations = {
        "Argentina": "Argentina",
        "Australia": "Australia",
        "Austria": "Austria",
        "Belgium": "Bélgica",
        "Brazil": "Brasil",
        "Canada": "Canadá",
        "China": "China",
        "France": "Francia",
        "Germany": "Alemania",
        "India": "India",
        "Iran": "Irán",
        "Israel": "Israel",
        "Italy": "Italia",
        "Japan": "Japón",
        "Kazakhstan": "Kazajistán",
        "Mexico": "México",
        "Netherlands": "Países Bajos",
        "Russia": "Rusia",
        "South Korea": "Corea del Sur",
        "Spain": "España",
        "Sweden": "Suecia",
        "Ukraine": "Ucrania",
        "United Kingdom": "Reino Unido",
        "United States": "Estados Unidos",
    }

    return translations.get(str(name), str(name))


def flag_url_from_iso3(code: object) -> str:
    if pd.isna(code):
        return ""

    iso3_to_iso2 = {
        "AFG": "af", "ALB": "al", "DZA": "dz", "AND": "ad", "AGO": "ao",
        "ARG": "ar", "ARM": "am", "AUS": "au", "AUT": "at", "AZE": "az",
        "BHR": "bh", "BGD": "bd", "BLR": "by", "BEL": "be", "BOL": "bo",
        "BRA": "br", "BGR": "bg", "CAN": "ca", "CHL": "cl", "CHN": "cn",
        "COL": "co", "CZE": "cz", "DNK": "dk", "ECU": "ec", "EGY": "eg",
        "EST": "ee", "FIN": "fi", "FRA": "fr", "DEU": "de", "GRC": "gr",
        "HUN": "hu", "IND": "in", "IDN": "id", "IRN": "ir", "IRQ": "iq",
        "IRL": "ie", "ISR": "il", "ITA": "it", "JPN": "jp", "JOR": "jo",
        "KAZ": "kz", "KEN": "ke", "KOR": "kr", "MYS": "my", "MEX": "mx",
        "NLD": "nl", "NZL": "nz", "NGA": "ng", "NOR": "no", "PAK": "pk",
        "PER": "pe", "PHL": "ph", "POL": "pl", "PRT": "pt", "ROU": "ro",
        "RUS": "ru", "SAU": "sa", "SGP": "sg", "ZAF": "za", "ESP": "es",
        "SWE": "se", "CHE": "ch", "THA": "th", "TUR": "tr", "UKR": "ua",
        "ARE": "ae", "GBR": "gb", "USA": "us", "VEN": "ve", "VNM": "vn",
    }

    iso2 = iso3_to_iso2.get(str(code).upper())
    if iso2 is None:
        return ""

    return f"https://public.flourish.studio/country-flags/svg/{iso2}.svg"


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    satcat = clean_columns(pd.read_csv(RAW_DIR / "satcat.csv"))
    ucs = clean_columns(pd.read_excel(RAW_DIR / "ucs_satellite_database_2023.xlsx"))
    owid = clean_columns(pd.read_csv(RAW_DIR / "owid_objects_launched.csv"))

    print("Datos cargados:")
    print(f"SATCAT: {satcat.shape}")
    print(f"UCS: {ucs.shape}")
    print(f"OWID: {owid.shape}")

    return satcat, ucs, owid


def prepare_satcat(satcat: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "object_name",
        "object_id",
        "norad_cat_id",
        "object_type",
        "ops_status_code",
        "owner",
        "launch_date",
        "launch_site",
        "decay_date",
        "period",
        "inclination",
        "apogee",
        "perigee",
        "rcs",
        "data_status_code",
        "orbit_center",
        "orbit_type",
    ]

    df = satcat[cols].copy()

    numeric_cols = ["norad_cat_id", "period", "inclination", "apogee", "perigee", "rcs"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    df["decay_date"] = pd.to_datetime(df["decay_date"], errors="coerce")

    df["launch_year"] = df["launch_date"].dt.year.astype("Int64")
    df["decay_year"] = df["decay_date"].dt.year.astype("Int64")
    df["altitude_mean_km"] = df[["apogee", "perigee"]].mean(axis=1).round(2)

    df["in_orbit_flag"] = df["decay_date"].isna()
    df["age_years"] = ((SNAPSHOT_DATE - df["launch_date"]).dt.days / 365.25).round(2)

    df["is_payload"] = df["object_type"].eq("PAY")
    df["is_debris"] = df["object_type"].eq("DEB")
    df["is_rocket_body"] = df["object_type"].eq("R/B")

    df["object_type_label"] = df["object_type"].map(
        {
            "PAY": "Payload / satélite",
            "DEB": "Debris / residuo",
            "R/B": "Cuerpo de cohete",
            "UNK": "Desconocido",
        }
    ).fillna("Otros")

    df["object_type_short"] = df["object_type"].map(
        {
            "PAY": "payload",
            "DEB": "debris",
            "R/B": "rocket_body",
            "UNK": "unknown",
        }
    ).fillna("other")

    df["launch_decade"] = (df["launch_year"] // 10 * 10).astype("Int64")
    df["launch_decade_label"] = df["launch_decade"].astype(str) + "s"

    return df


def prepare_ucs(ucs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "current_official_name_of_satellite",
        "country_org_of_un_registry",
        "country_of_operator_owner",
        "operator_owner",
        "users",
        "purpose",
        "detailed_purpose",
        "class_of_orbit",
        "type_of_orbit",
        "longitude_of_geo_degrees",
        "perigee_km",
        "apogee_km",
        "eccentricity",
        "inclination_degrees",
        "period_minutes",
        "launch_mass_kg",
        "dry_mass_kg",
        "power_watts",
        "date_of_launch",
        "expected_lifetime_yrs",
        "contractor",
        "country_of_contractor",
        "launch_site",
        "launch_vehicle",
        "cospar_number",
        "norad_number",
        "comments",
    ]

    df = ucs[cols].copy()
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    numeric_cols = [
        "longitude_of_geo_degrees",
        "perigee_km",
        "apogee_km",
        "eccentricity",
        "inclination_degrees",
        "period_minutes",
        "launch_mass_kg",
        "dry_mass_kg",
        "power_watts",
        "expected_lifetime_yrs",
        "norad_number",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date_of_launch"] = pd.to_datetime(df["date_of_launch"], errors="coerce")
    df["completeness_score"] = df.notna().sum(axis=1)

    duplicates = df[df.duplicated(subset="norad_number", keep=False)].copy()
    duplicates.to_csv(INTERIM_DIR / "ucs_duplicate_keys_review.csv", index=False, encoding="utf-8-sig")

    df = df.sort_values(
        by=["norad_number", "completeness_score", "date_of_launch"],
        ascending=[True, False, False],
    )

    df = df.drop_duplicates(subset="norad_number", keep="first").copy()
    df = df.drop(columns=["completeness_score"])

    rename_map = {
        "current_official_name_of_satellite": "ucs_official_name",
        "country_org_of_un_registry": "un_registry_country",
        "country_of_operator_owner": "operator_owner_country",
        "class_of_orbit": "ucs_orbit_class",
        "type_of_orbit": "ucs_orbit_type",
        "longitude_of_geo_degrees": "geo_longitude_degrees",
        "perigee_km": "ucs_perigee_km",
        "apogee_km": "ucs_apogee_km",
        "inclination_degrees": "ucs_inclination_degrees",
        "period_minutes": "ucs_period_minutes",
        "date_of_launch": "ucs_launch_date",
        "launch_site": "ucs_launch_site",
        "norad_number": "norad_cat_id",
    }

    df = df.rename(columns=rename_map)

    return df


def add_altitude_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conditions = [
        df["altitude_mean_km"].lt(400),
        df["altitude_mean_km"].between(400, 600, inclusive="left"),
        df["altitude_mean_km"].between(600, 1200, inclusive="left"),
        df["altitude_mean_km"].between(1200, 2000, inclusive="left"),
        df["altitude_mean_km"].between(2000, 20000, inclusive="left"),
        df["altitude_mean_km"].between(20000, 30000, inclusive="left"),
        df["altitude_mean_km"].between(30000, 40000, inclusive="left"),
        df["altitude_mean_km"].ge(40000),
    ]

    labels = [
        "VLEO (<400 km)",
        "LEO baja (400-600 km)",
        "LEO media (600-1200 km)",
        "LEO alta (1200-2000 km)",
        "MEO (2000-20000 km)",
        "MEO alta / GNSS (20000-30000 km)",
        "GEO aprox. (30000-40000 km)",
        "Órbita alta / HEO (>40000 km)",
    ]

    order = {label: i + 1 for i, label in enumerate(labels)}

    df["altitude_band"] = np.select(conditions, labels, default="Sin altitud válida")
    df["altitude_band_order"] = df["altitude_band"].map(order).fillna(99).astype(int)

    return df


def simplify_users(value: object) -> str:
    if pd.isna(value):
        return "Sin clasificar"

    text = str(value).lower()

    if "commercial" in text:
        return "Comercial"
    if "military" in text:
        return "Militar"
    if "government" in text:
        return "Gubernamental"
    if "civil" in text:
        return "Civil"

    return "Otros"


def simplify_purpose(value: object) -> str:
    if pd.isna(value):
        return "Sin clasificar"

    text = str(value).lower()

    if "communication" in text:
        return "Comunicaciones"
    if "earth observation" in text or "remote sensing" in text:
        return "Observación terrestre"
    if "technology" in text:
        return "Desarrollo tecnológico"
    if "navigation" in text or "positioning" in text:
        return "Navegación / posicionamiento"
    if "space science" in text or "science" in text:
        return "Ciencia espacial"
    if "meteorology" in text or "weather" in text:
        return "Meteorología"
    if "surveillance" in text or "reconnaissance" in text:
        return "Vigilancia / reconocimiento"

    return "Otros"


def build_merged_dataset(satcat_clean: pd.DataFrame, ucs_clean: pd.DataFrame) -> pd.DataFrame:
    df = satcat_clean.merge(
        ucs_clean,
        on="norad_cat_id",
        how="left",
        validate="one_to_one",
    ).copy()

    df["matched_ucs"] = df["ucs_official_name"].notna()
    df["users_simplified"] = df["users"].apply(simplify_users)
    df["purpose_simplified"] = df["purpose"].apply(simplify_purpose)

    df = add_altitude_bands(df)

    df.to_csv(PROCESSED_DIR / "objects_orbit_enriched.csv", index=False, encoding="utf-8-sig")

    payloads = df[df["object_type"] == "PAY"].copy()
    payloads.to_csv(PROCESSED_DIR / "payload_objects_enriched.csv", index=False, encoding="utf-8-sig")

    payload_semantic = payloads[payloads["matched_ucs"]].copy()
    payload_semantic.to_csv(PROCESSED_DIR / "payload_semantic_enriched.csv", index=False, encoding="utf-8-sig")

    print("Dataset enriquecido:")
    print(f"objects_orbit_enriched: {df.shape}")
    print(f"payload_objects_enriched: {payloads.shape}")
    print(f"payload_semantic_enriched: {payload_semantic.shape}")
    print(f"Match UCS global: {df['matched_ucs'].mean():.2%}")
    print(f"Match UCS en PAY: {payloads['matched_ucs'].mean():.2%}")

    return df


def build_kpi_intro(objects: pd.DataFrame) -> None:
    in_orbit = objects[objects["in_orbit_flag"]].copy()

    rows = [
        {
            "metric_id": "total_objects",
            "label": "Objetos registrados",
            "value": len(objects),
            "unit": "objetos",
            "description": "Total de objetos orbitales registrados en SATCAT.",
        },
        {
            "metric_id": "objects_in_orbit",
            "label": "Objetos sin fecha de decaimiento",
            "value": len(in_orbit),
            "unit": "objetos",
            "description": "Objetos que permanecen en órbita según la ausencia de fecha de decaimiento.",
        },
        {
            "metric_id": "payloads_in_orbit",
            "label": "Cargas útiles en órbita",
            "value": int(((objects["object_type"] == "PAY") & objects["in_orbit_flag"]).sum()),
            "unit": "objetos",
            "description": "Satélites o cargas útiles sin fecha de decaimiento registrada.",
        },
        {
            "metric_id": "debris_in_orbit",
            "label": "Residuos en órbita",
            "value": int(((objects["object_type"] == "DEB") & objects["in_orbit_flag"]).sum()),
            "unit": "objetos",
            "description": "Fragmentos o debris sin fecha de decaimiento registrada.",
        },
    ]

    save_csv(pd.DataFrame(rows), "00_kpi_intro.csv")


def build_owid_datasets(owid: pd.DataFrame) -> None:
    df = owid.copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["annual_launches"] = pd.to_numeric(df["annual_launches"], errors="coerce")
    df = df.dropna(subset=["year", "annual_launches"]).copy()
    df = df[df["year"] <= MAX_COMPLETE_YEAR].copy()

    country_year = df[
        df["code"].notna()
        & df["code"].astype(str).str.len().eq(3)
        & ~df["entity"].str.lower().eq("world")
    ].copy()

    country_year = country_year.sort_values(["entity", "year"])
    country_year["cumulative_launches"] = (
        country_year.groupby("entity")["annual_launches"].cumsum()
    )

    save_csv(
        country_year[["entity", "code", "year", "annual_launches", "cumulative_launches"]],
        "01_owid_country_year.csv",
    )

    # Cada país aparece en todos los años de 1957 a MAX_COMPLETE_YEAR.
    years = pd.DataFrame({"year": range(1957, MAX_COMPLETE_YEAR + 1)})

    countries = (
        country_year[["entity", "code"]]
        .drop_duplicates()
        .sort_values(["entity", "code"])
        .reset_index(drop=True)
    )

    country_year_complete = (
        countries.assign(key=1)
        .merge(years.assign(key=1), on="key")
        .drop(columns="key")
        .merge(
            country_year[["entity", "code", "year", "annual_launches"]],
            on=["entity", "code", "year"],
            how="left",
        )
    )

    country_year_complete["annual_launches"] = (
        country_year_complete["annual_launches"]
        .fillna(0)
        .astype(int)
    )

    country_year_complete = country_year_complete.sort_values(["entity", "year"]).copy()

    country_year_complete["cumulative_launches"] = (
        country_year_complete
        .groupby("entity")["annual_launches"]
        .cumsum()
    )

    save_csv(
        country_year_complete[
            ["entity", "code", "year", "annual_launches", "cumulative_launches"]
        ],
        "01_owid_country_year_complete.csv",
    )

    country_race_base = country_year_complete.copy()

    country_totals = (
        country_race_base.groupby(["entity", "code"], as_index=False)["cumulative_launches"]
        .max()
        .sort_values("cumulative_launches", ascending=False)
    )

    top_countries = country_totals.head(20)[["entity", "code"]]

    country_race_base = country_race_base.merge(
        top_countries,
        on=["entity", "code"],
        how="inner",
    )

    country_race_wide = (
        country_race_base.pivot_table(
            index=["entity", "code"],
            columns="year",
            values="cumulative_launches",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    country_race_wide.columns = country_race_wide.columns.astype(str)

    country_race_wide = country_race_wide.sort_values(
        str(MAX_COMPLETE_YEAR),
        ascending=False,
    )

    country_race_wide.insert(1, "category", "País")
    country_race_wide.insert(
        2,
        "image_url",
        country_race_wide["code"].apply(flag_url_from_iso3),
    )

    country_race_wide["entity"] = country_race_wide["entity"].apply(translate_country_name)

    country_race_wide = country_race_wide.rename(columns={"entity": "country"})
    country_race_wide = country_race_wide.drop(columns=["code"])

    save_csv(country_race_wide, "01_owid_country_cumulative_race.csv")

    world = df[df["entity"].str.lower().eq("world")].copy()

    if world.empty:
        world = (
            country_year.groupby("year", as_index=False)["annual_launches"]
            .sum()
            .assign(entity="World")
        )

    world = world.sort_values("year").copy()
    world["cumulative_launches"] = world["annual_launches"].cumsum()

    save_csv(
        world[["entity", "year", "annual_launches", "cumulative_launches"]],
        "01_owid_global_year.csv",
    )


def build_objects_by_type_year(objects: pd.DataFrame) -> None:
    df = objects.dropna(subset=["launch_year"]).copy()
    df["launch_year"] = df["launch_year"].astype(int)
    df = df[df["launch_year"] <= MAX_COMPLETE_YEAR].copy()

    long = (
        df.groupby(["launch_year", "object_type_short", "object_type_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_objects", "launch_year": "year"})
        .sort_values(["year", "object_type_short"])
    )

    wide = (
        long.pivot_table(
            index="year",
            columns="object_type_short",
            values="n_objects",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["payload", "debris", "rocket_body", "unknown", "other"]:
        if col not in wide.columns:
            wide[col] = 0

    wide["total"] = wide[["payload", "debris", "rocket_body", "unknown", "other"]].sum(axis=1)
    wide = wide[["year", "payload", "debris", "rocket_body", "unknown", "other", "total"]]

    save_csv(wide, "02_objects_by_type_year_wide.csv")
    save_csv(long, "02_objects_by_type_year_long.csv")


def build_orbital_band_datasets(objects: pd.DataFrame) -> None:
    current_with_altitude = objects[
        objects["in_orbit_flag"]
        & objects["altitude_mean_km"].notna()
    ].copy()

    current_for_heatmap = current_with_altitude[
        current_with_altitude["launch_decade"].notna()
        & current_with_altitude["launch_year"].notna()
    ].copy()

    current_for_heatmap["launch_year"] = current_for_heatmap["launch_year"].astype(int)
    current_for_heatmap = current_for_heatmap[
        current_for_heatmap["launch_year"] <= MAX_COMPLETE_YEAR
    ].copy()

    current_for_heatmap["launch_decade"] = current_for_heatmap["launch_decade"].astype(int)
    current_for_heatmap["launch_decade_label"] = current_for_heatmap["launch_decade"].astype(str) + "s"

    current_for_heatmap.loc[
        current_for_heatmap["launch_decade"] == 2020,
        "launch_decade_label"
    ] = "2020-2025"

    heatmap_long = (
        current_for_heatmap.groupby(
            ["altitude_band_order", "altitude_band", "launch_decade_label"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_objects"})
    )

    heatmap = (
        heatmap_long.pivot_table(
            index=["altitude_band_order", "altitude_band"],
            columns="launch_decade_label",
            values="n_objects",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .sort_values("altitude_band_order")
    )

    heatmap = heatmap.drop(columns=["altitude_band_order"])

    summary = (
        current_with_altitude.groupby(["altitude_band_order", "altitude_band"], as_index=False)
        .size()
        .rename(columns={"size": "n_objects"})
        .sort_values("altitude_band_order")
    )

    total = summary["n_objects"].sum()
    summary["share_pct"] = (summary["n_objects"] / total * 100).round(2)

    save_csv(heatmap, "03_orbital_band_decade_heatmap.csv")
    save_csv(summary, "03_current_objects_by_altitude_band.csv")
    orbital_band_decade_long = heatmap.melt(
        id_vars="altitude_band",
        var_name="decade",
        value_name="objects"
    )

    save_csv(
        orbital_band_decade_long,
        FLOURISH_DIR / "03_orbital_band_decade_heatmap_long.csv"
    )


def build_payload_user_datasets(objects: pd.DataFrame) -> None:
    payloads = objects[
        (objects["object_type"] == "PAY")
        & objects["matched_ucs"]
        & objects["in_orbit_flag"]
        & objects["launch_year"].notna()
    ].copy()

    payloads["year"] = payloads["launch_year"].astype(int)
    payloads = payloads[payloads["year"] <= MAX_COMPLETE_YEAR].copy()

    long = (
        payloads.groupby(["year", "users_simplified"], as_index=False)
        .size()
        .rename(columns={"size": "n_payloads"})
        .sort_values(["year", "users_simplified"])
    )

    wide = (
        long.pivot_table(
            index="year",
            columns="users_simplified",
            values="n_payloads",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    expected_cols = ["Comercial", "Militar", "Gubernamental", "Civil", "Otros", "Sin clasificar"]
    for col in expected_cols:
        if col not in wide.columns:
            wide[col] = 0

    wide["total"] = wide[expected_cols].sum(axis=1)
    wide = wide[["year"] + expected_cols + ["total"]].sort_values("year")

    commercial = wide[["year", "Comercial"]].copy()
    commercial = commercial.rename(columns={"Comercial": "commercial_payloads"})
    commercial["previous_year_payloads"] = commercial["commercial_payloads"].shift(1)
    commercial["commercial_growth_pct"] = np.where(
        commercial["previous_year_payloads"] > 0,
        (
            (commercial["commercial_payloads"] - commercial["previous_year_payloads"])
            / commercial["previous_year_payloads"]
            * 100
        ),
        np.nan,
    ).round(2)
    commercial["commercial_payloads_3yr_avg"] = (
        commercial["commercial_payloads"].rolling(window=3, min_periods=1).mean().round(2)
    )

    wide_since_2000 = wide[wide["year"] >= 2000].copy()
    long_since_2000 = long[long["year"] >= 2000].copy()
    commercial_since_2000 = commercial[commercial["year"] >= 2000].copy()

    save_csv(wide, "04_payloads_by_user_year_wide.csv")
    save_csv(long, "04_payloads_by_user_year_long.csv")
    save_csv(commercial, "04_commercial_growth_rate.csv")

    save_csv(wide_since_2000, "04_payloads_by_user_year_since_2000_wide.csv")
    save_csv(long_since_2000, "04_payloads_by_user_year_since_2000_long.csv")
    save_csv(commercial_since_2000, "04_commercial_growth_rate_since_2000.csv")


def build_operator_concentration(objects: pd.DataFrame) -> None:
    payloads = objects[
        (objects["object_type"] == "PAY")
        & objects["matched_ucs"]
        & objects["in_orbit_flag"]
        & objects["operator_owner"].notna()
    ].copy()

    grouped = (
        payloads.groupby(["operator_owner_country", "operator_owner"], as_index=False)
        .size()
        .rename(columns={"size": "n_satellites"})
        .sort_values("n_satellites", ascending=False)
    )

    total = grouped["n_satellites"].sum()
    grouped["share_pct"] = (grouped["n_satellites"] / total * 100).round(3)

    top_n = 30
    top = grouped.head(top_n).copy()
    rest = grouped.iloc[top_n:].copy()

    if not rest.empty:
        rest_row = pd.DataFrame(
            [
                {
                    "operator_owner_country": "Otros",
                    "operator_owner": "Otros operadores",
                    "n_satellites": rest["n_satellites"].sum(),
                    "share_pct": round(rest["n_satellites"].sum() / total * 100, 3),
                }
            ]
        )
        grouped_final = pd.concat([top, rest_row], ignore_index=True)
    else:
        grouped_final = top

    grouped_final = grouped_final.rename(
        columns={
            "operator_owner_country": "country",
            "operator_owner": "operator",
        }
    )

    save_csv(grouped_final, "05_operator_concentration.csv")


def build_purpose_orbit_flow(objects: pd.DataFrame) -> None:
    payloads = objects[
        (objects["object_type"] == "PAY")
        & objects["matched_ucs"]
        & objects["in_orbit_flag"]
        & objects["purpose_simplified"].notna()
        & objects["ucs_orbit_class"].notna()
    ].copy()

    flow = (
        payloads.groupby(["purpose_simplified", "ucs_orbit_class"], as_index=False)
        .size()
        .rename(
            columns={
                "purpose_simplified": "source",
                "ucs_orbit_class": "target",
                "size": "value",
            }
        )
        .sort_values("value", ascending=False)
    )

    save_csv(flow, "06_purpose_orbit_flow.csv")


def format_integer_es(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def format_percent_es(value: float) -> str:
    if pd.isna(value):
        return ""

    return f"{value:.1f}".replace(".", ",") + "%"


def build_sustainability_indicators(objects: pd.DataFrame) -> None:
    current = objects[objects["in_orbit_flag"]].copy()
    current_with_altitude = current[current["altitude_mean_km"].notna()].copy()

    payload_semantic_current = current[
        (current["object_type"] == "PAY")
        & current["matched_ucs"]
    ].copy()

    debris_in_orbit = int((current["object_type"] == "DEB").sum())
    rocket_bodies_in_orbit = int((current["object_type"] == "R/B").sum())
    payloads_in_orbit = int((current["object_type"] == "PAY").sum())

    leo_current = current_with_altitude[current_with_altitude["altitude_mean_km"] < 2000]
    leo_share = (
        len(leo_current) / len(current_with_altitude) * 100
        if len(current_with_altitude)
        else np.nan
    )

    commercial_payloads = payload_semantic_current[
        payload_semantic_current["users_simplified"] == "Comercial"
    ]

    commercial_share = (
        len(commercial_payloads) / len(payload_semantic_current) * 100
        if len(payload_semantic_current)
        else np.nan
    )

    operator_counts = (
        payload_semantic_current.dropna(subset=["operator_owner"])
        .groupby("operator_owner")
        .size()
        .sort_values(ascending=False)
    )

    top10_share = (
        operator_counts.head(10).sum() / operator_counts.sum() * 100
        if operator_counts.sum() > 0
        else np.nan
    )

    launched_recent = objects[
        objects["launch_year"].notna()
        & objects["launch_year"].between(2020, MAX_COMPLETE_YEAR, inclusive="both")
    ]

    recent_launches = len(launched_recent)

    rows = [
        {
            "metric_id": "objects_in_orbit",
            "title": "Objetos en órbita",
            "value": len(current),
            "value_display": format_integer_es(len(current)),
            "description_short": "Objetos catalogados sin fecha de decaimiento registrada.",
            "category": "Presión orbital",
            "sort_order": 1,
        },
        {
            "metric_id": "payloads_in_orbit",
            "title": "Cargas útiles",
            "value": payloads_in_orbit,
            "value_display": format_integer_es(payloads_in_orbit),
            "description_short": "Satélites y cargas útiles que permanecen catalogados en órbita.",
            "category": "Uso orbital",
            "sort_order": 2,
        },
        {
            "metric_id": "debris_and_rocket_bodies",
            "title": "Residuos y cohetes",
            "value": debris_in_orbit + rocket_bodies_in_orbit,
            "value_display": format_integer_es(debris_in_orbit + rocket_bodies_in_orbit),
            "description_short": "Debris y cuerpos de cohete sin fecha de decaimiento registrada.",
            "category": "Sostenibilidad",
            "sort_order": 3,
        },
        {
            "metric_id": "leo_share",
            "title": "Peso de la órbita baja",
            "value": round(leo_share, 2),
            "value_display": format_percent_es(leo_share),
            "description_short": "Porcentaje de objetos en órbita con altitud media inferior a 2.000 km.",
            "category": "Concentración orbital",
            "sort_order": 4,
        },
        {
            "metric_id": "commercial_payload_share",
            "title": "Peso comercial",
            "value": round(commercial_share, 2),
            "value_display": format_percent_es(commercial_share),
            "description_short": "Peso de los payloads comerciales dentro del subconjunto enriquecido.",
            "category": "Mercado espacial",
            "sort_order": 5,
        },
        {
            "metric_id": "top10_operator_share",
            "title": "Top 10 operadores",
            "value": round(top10_share, 2),
            "value_display": format_percent_es(top10_share),
            "description_short": "Concentración de cargas útiles enriquecidas en los diez principales operadores.",
            "category": "Concentración",
            "sort_order": 6,
        },
        {
            "metric_id": "recent_launches",
            "title": "Lanzamientos recientes",
            "value": recent_launches,
            "value_display": format_integer_es(recent_launches),
            "description_short": "Objetos catalogados lanzados entre 2020 y 2025.",
            "category": "Aceleración",
            "sort_order": 7,
        },
    ]

    cards = pd.DataFrame(rows).sort_values("sort_order").copy()

    save_csv(cards, "07_sustainability_indicators.csv")

def main() -> None:
    satcat, ucs, owid = load_raw_data()

    satcat_clean = prepare_satcat(satcat)
    ucs_clean = prepare_ucs(ucs)

    satcat_clean.to_csv(INTERIM_DIR / "satcat_clean.csv", index=False, encoding="utf-8-sig")
    ucs_clean.to_csv(INTERIM_DIR / "ucs_clean.csv", index=False, encoding="utf-8-sig")

    objects = build_merged_dataset(satcat_clean, ucs_clean)

    build_kpi_intro(objects)
    build_owid_datasets(owid)
    build_objects_by_type_year(objects)
    build_orbital_band_datasets(objects)
    build_payload_user_datasets(objects)
    build_operator_concentration(objects)
    build_purpose_orbit_flow(objects)
    build_sustainability_indicators(objects)

    print("Proceso completado.")


if __name__ == "__main__":
    main()