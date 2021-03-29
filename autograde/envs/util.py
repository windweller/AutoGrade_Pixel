import re
import imageio
import base64

# Used in All 4
def jpg_b64_to_rgb(state):
    return imageio.imread(bytes(base64.b64decode(state)))

# Used in Color
def getRGBA(bgcolor):
    m = re.fullmatch(r"rgba\((.*),(.*),(.*),(.*)\)", bgcolor)
    return [float(m.group(i)) for i in range(1,5)]
