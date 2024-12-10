import sys
import os
import time
import array
sys.path.append(os.getcwd())

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation
import tqdm
import json

# load pyrenderer
from diffdvr import make_real3
from diffdvr.settings import Settings
import pyrenderer

from vis import tfvis


if __name__=='__main__':
    print(pyrenderer.__doc__)
    files = [
        ('config-files/tooth1.json', 'Tooth1.xml'),
        ('config-files/tooth3gauss.json', 'Tooth3Gauss.xml'),
        ('config-files/tooth3linear.json', 'Tooth3Linear.xml'),
    ]
    for input_path, output_path in files:
        settings = Settings(input_path)
        tf_points = settings.get_tf_points(opacity_scaling=1.0)

        with open(output_path, 'w') as out_file:
            out_file.write('<TransferFunction colorspace="sRGB" interpolation_colorspace="Linear RGB">\n')
            out_file.write('    <OpacityPoints>\n')
            for tf_point in tf_points:
                pos = np.clip(tf_point.pos, 0.0, 1.0)
                out_file.write(f'        <OpacityPoint position="{pos}" opacity="{tf_point.val.w}"/>\n')
            out_file.write('    </OpacityPoints>\n')
            out_file.write('    <ColorPoints color_data="float">\n')
            for tf_point in tf_points:
                pos = np.clip(tf_point.pos, 0.0, 1.0)
                out_file.write(f'        <ColorPoint position="{pos}" r="{tf_point.val.x}" g="{tf_point.val.y}" b="{tf_point.val.z}"/>\n')
            out_file.write('    </ColorPoints>\n')
            out_file.write('</TransferFunction>\n')

    pyrenderer.cleanup()
