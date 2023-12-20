import distinctipy
N = 100
_colors = distinctipy.get_colors(100)
COLORS = []
for _color in _colors:
    color = []
    for c in _color:
        color.append(c*255)
    COLORS.append(color) 