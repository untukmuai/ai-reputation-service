

def get_average_hex_color(pil_image):
    avg_color = pil_image.resize((1, 1)).getpixel((0, 0))
    if len(avg_color) == 4:
        avg_color = avg_color[:3]
    return '#{:02x}{:02x}{:02x}'.format(*avg_color)