DAY_NIGHT_MAP = {
    "clear":      0, "fog":        0, "for_rain":   0,
    "rain":       0, "snow":       0, "night":      1,
    "night_fog":  1, "night_rain": 1, "night_snow": 1,
}

WEATHER_TYPE_MAP = {
    "clear":      0, "fog":        1, "for_rain":   2,
    "rain":       3, "snow":       4, "night":      0,
    "night_fog":  1, "night_rain": 3, "night_snow": 4,
}

COMBO_TO_FINAL = {
    (0, 0): 0,  # clear
    (0, 1): 1,  # fog
    (0, 2): 2,  # for_rain
    (1, 2): 2,  # night + fog_rain -> for_rain
    (0, 3): 7,  # rain
    (0, 4): 8,  # snow
    (1, 0): 3,  # night
    (1, 1): 4,  # night_fog
    (1, 3): 5,  # night_rain
    (1, 4): 6,  # night_snow
}
