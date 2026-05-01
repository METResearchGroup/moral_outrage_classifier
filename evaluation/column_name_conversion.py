COLUMN_NAME_CONVERSION: dict[str, list[str]] = {
    "id": ["id", "tweet_id"],
    "text": ["text", "body"],
    "gold_label": ["gold_label", "outrage", "pers_outrage_label"],
}

REQUIRED_COLUMNS = ("id", "text")
