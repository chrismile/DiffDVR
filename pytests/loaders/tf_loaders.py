#
# BSD 2-Clause License
#
# Copyright (c) 2022, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from enum import Enum
import xml.etree.ElementTree as ET


class ColorDataMode(Enum):
    UNSIGNED_BYTE = 0
    UNSIGNED_SHORT = 1
    FLOAT_NORMALIZED = 2
    FLOAT_255 = 3


COLOR_DATA_MODE_NAMES = ['ubyte', 'ushort', 'float', 'float_255']


def srgb_to_linearrgb(val):
    # See https://en.wikipedia.org/wiki/SRGB
    if val <= 0.04045:
        return val / 12.92
    else:
        return pow((val + 0.055) / 1.055, 2.4)


def linearrgb_to_srgb(val):
    # See https://en.wikipedia.org/wiki/SRGB
    if val <= 0.0031308:
        return val * 12.92
    else:
        return 1.055 * pow(val, 1.0 / 2.4) - 0.055


def load_tf_from_xml(path, res, opacity_scale, write_pos=False):
    root = ET.parse(path).getroot()

    interpolation_colorspace = root.get('interpolation_colorspace')

    color_points_node = root.find('ColorPoints')
    color_data_mode = ColorDataMode.UNSIGNED_BYTE
    color_data_mode_string = color_points_node.get('color_data')
    if color_data_mode_string is not None:
        for i, name in enumerate(COLOR_DATA_MODE_NAMES):
            if color_data_mode_string == name:
                color_data_mode = ColorDataMode(i)

    opacity_points = []
    for opacity_point_node in root.findall('OpacityPoints/OpacityPoint'):
        position = float(opacity_point_node.get('position'))
        opacity = float(opacity_point_node.get('opacity')) * opacity_scale
        opacity_points.append({'position': position, 'opacity': opacity})
    color_points = []
    for color_point_node in root.findall('ColorPoints/ColorPoint'):
        position = float(color_point_node.get('position'))
        r = float(color_point_node.get('r'))
        g = float(color_point_node.get('g'))
        b = float(color_point_node.get('b'))
        if color_data_mode == ColorDataMode.UNSIGNED_BYTE or color_data_mode == ColorDataMode.FLOAT_255:
            r /= 255.0
            g /= 255.0
            b /= 255.0
        elif color_data_mode == ColorDataMode.UNSIGNED_SHORT:
            r /= 65535.0
            g /= 65535.0
            b /= 65535.0
        color_points.append({'position': position, 'r': r, 'g': g, 'b': b})

    tf_entries = []
    color_pt_idx = 0
    opacity_pt_idx = 0
    for i in range(res):
        curr_pos = i / (res - 1)
        while color_points[color_pt_idx]['position'] < curr_pos:
            color_pt_idx += 1
        while opacity_points[opacity_pt_idx]['position'] < curr_pos:
            opacity_pt_idx += 1

        if write_pos:
            color_at_idx = [0.0, 0.0, 0.0, 0.0, curr_pos]  # r,g,b,a,pos
        else:
            color_at_idx = [0.0, 0.0, 0.0, 0.0]  # r,g,b,a

        if color_points[color_pt_idx]['position'] == curr_pos:
            color_at_idx[0] = color_points[color_pt_idx]['r']
            color_at_idx[1] = color_points[color_pt_idx]['g']
            color_at_idx[2] = color_points[color_pt_idx]['b']
        else:
            pos0 = color_points[color_pt_idx - 1]['position']
            pos1 = color_points[color_pt_idx]['position']
            f0 = (pos1 - curr_pos) / (pos1 - pos0)
            f1 = 1.0 - f0
            c0r_srgb = color_points[color_pt_idx - 1]['r']
            c0g_srgb = color_points[color_pt_idx - 1]['g']
            c0b_srgb = color_points[color_pt_idx - 1]['b']
            c1r_srgb = color_points[color_pt_idx]['r']
            c1g_srgb = color_points[color_pt_idx]['g']
            c1b_srgb = color_points[color_pt_idx]['b']
            if interpolation_colorspace == 'Linear RGB':
                c0r = srgb_to_linearrgb(c0r_srgb)
                c0g = srgb_to_linearrgb(c0g_srgb)
                c0b = srgb_to_linearrgb(c0b_srgb)
                c1r = srgb_to_linearrgb(c1r_srgb)
                c1g = srgb_to_linearrgb(c1g_srgb)
                c1b = srgb_to_linearrgb(c1b_srgb)
            else:
                c0r = c0r_srgb
                c0g = c0g_srgb
                c0b = c0b_srgb
                c1r = c1r_srgb
                c1g = c1g_srgb
                c1b = c1b_srgb
            cr = f0 * c0r + f1 * c1r
            cg = f0 * c0g + f1 * c1g
            cb = f0 * c0b + f1 * c1b
            if interpolation_colorspace == 'Linear RGB':
                cr = linearrgb_to_srgb(cr)
                cg = linearrgb_to_srgb(cg)
                cb = linearrgb_to_srgb(cb)
            color_at_idx[0] = cr
            color_at_idx[1] = cg
            color_at_idx[2] = cb

        if opacity_points[opacity_pt_idx]['position'] == curr_pos:
            color_at_idx[3] = opacity_points[opacity_pt_idx]['opacity']
        else:
            pos0 = opacity_points[opacity_pt_idx - 1]['position']
            pos1 = opacity_points[opacity_pt_idx]['position']
            f0 = (pos1 - curr_pos) / (pos1 - pos0)
            f1 = 1.0 - f0
            opacity0 = opacity_points[opacity_pt_idx - 1]['opacity']
            opacity1 = opacity_points[opacity_pt_idx]['opacity']
            color_at_idx[3] = f0 * opacity0 + f1 * opacity1

        tf_entries.append(color_at_idx)

    return tf_entries

