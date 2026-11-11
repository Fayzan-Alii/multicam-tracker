import colorsys


def id_color(identity: int) -> tuple[int, int, int]:
    """Distinct, stable BGR colour per identity (golden-ratio hue spacing)."""
    hue = (identity * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)
