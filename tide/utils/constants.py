COUNTRY_CODES = [
    "USA", "UK", "JP", "FR", "DE", "NL", "KY", "LI", "MC", "BS", "IM", "AE"
]

# Map to faker locales
COUNTRY_TO_LOCALE = {
    "USA": "en_US",
    "UK": "en_GB",
    "JP": "ja_JP",
    "FR": "fr_FR",
    "DE": "de_DE",
    "NL": "nl_NL",
    "KY": "en_US",
    "LI": "de_DE",
    "MC": "fr_FR",
    "BS": "en_US",
    "IM": "en_GB",
    "AE": "ar_AE",
}


COUNTRY_TO_CURRENCY = {
    "USA": "USD",
    "UK": "GBP",
    "JP": "JPY",
    "FR": "EUR",
    "DE": "EUR",
    "NL": "EUR",
    "KY": "USD",
    "LI": "CHF",
    "MC": "EUR",
    "BS": "BSD",
    "AE": "AED",
    "IM": "GBP",
}

HIGH_RISK_BUSINESS_CATEGORIES = [
    "Casinos",
    "Money Service Businesses",
    "Precious Metals Dealers",
    "Pawn Shops",
    "Check Cashing Services",
    "Currency Exchange",
    "Virtual Currency Exchange",
    "Art and Antiquities Dealers",
    "Jewelry Stores",
    "Convenience Stores",
    "Gas Stations",
    "Laundromats",
    "Bars and Nightclubs",
    "Auto Dealerships",
    "Used Car Sales",
    "Scrap Metal Dealers",
    "Tobacco Shops",
    "Liquor Stores",
    "Adult Entertainment",
    "Gun Shops",
    "Private Banking",
    "Investment Banking",
    "Trust Services",
]

HIGH_RISK_COUNTRIES = [
    "KY",  # Cayman Islands
    "BS",  # Bahamas
    "CH",  # Switzerland
    "MC",  # Monaco
    "CY",  # Cyprus
    "MT",  # Malta
    "SC",  # Seychelles
    "BB",  # Barbados
    "BM",  # Bermuda
    "AE",  # UAE
    "HK",  # Hong Kong
    "SG",  # Singapore
    "BZ",  # Belize
    "VU",  # Vanuatu
    "AG",  # Antigua and Barbuda
]

HIGH_RISK_OCCUPATIONS = [
    "Banker",
    "Retail banker",
    "Financial adviser",
    "Risk analyst",
    "Risk manager",
    "Corporate investment banker",
    "Investment banker, corporate",
    "Investment banker, operational",
    "Financial trader",

    "Lawyer",
    "Solicitor",
    "Licensed conveyancer",
    "Chartered accountant",
    "Chartered certified accountant",
    "Tax adviser",
    "Tax inspector",

    "Data processing manager",
    "Administrator",
    "Personal assistant",
    "IT consultant",

    "Restaurant manager",
    "Restaurant manager, fast food",
    "Public house manager",
    "Retail manager",
    "Dealer",

    "Estate agent",
    "Jewellery designer",

    "Cabin crew",
    "Tour manager",
    "Youth worker",
    "Community development worker",
    "Student"
]

HIGH_RISK_AGE_GROUPS = [
    "EIGHTEEN_TO_TWENTY_FOUR",
    "SIXTY_FIVE_PLUS"
]
