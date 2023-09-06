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

import os
import numpy as np
# conda install -c conda-forge nibabel
import nibabel as nib


def get_path_to_file(path):
    for i in reversed(range(len(path))):
        if path[i] == '/' or path[i] == '\\':
            return path[0:i+1]
    return path


def get_is_absolute_path(path):
    if os.name == 'nt':
        return (len(path) > 1 and path[1] == ':') or path.startswith('/') or path.startswith('\\')
    else:
        return path.startswith('/')


def load_dat_raw(file_path):
    dat_file_path = None
    raw_file_paths = []
    if file_path.endswith('.dat'):
        dat_file_path = file_path
    else:
        raw_file_paths.append(file_path)

        # We need to find the corresponding .dat file.
        raw_file_directory = get_path_to_file(file_path)
        for file in os.listdir(raw_file_directory):
             file_name = os.fsdecode(file)
             if file_name.endswith('.dat'):
                 dat_file_path = file_name
                 break
        if dat_file_path is None:
            raise Exception('Error in load_dat_raw: No .dat file found for "' + file_path + '".')

    dat_dict = {}
    with open(dat_file_path) as file:
        for line in file:
            split_line_string = line.rstrip().split(':')
            if len(split_line_string) == 0:
                continue
            elif len(split_line_string) != 2:
                raise Exception('Error in load_dat_raw: Invalid entry in file "' + dat_file_path + '".')
            key = split_line_string[0].strip().lower()
            value = split_line_string[1].strip()
            dat_dict[key] = value

    # Next, process the metadata.
    time_steps = []
    if len(raw_file_paths) == 0:
        if 'objectfilename' not in dat_dict:
            raise Exception('Error in load_dat_raw: Entry \'ObjectFileName\' missing in "' + dat_file_path + '".')
        objectfilename = dat_dict['objectfilename']
        if 'objectindices' in dat_dict:
            indices_split = dat_dict['objectindices'].split()
            start = int(indices_split[0])
            stop = int(indices_split[1])
            if len(indices_split) == 2:
                step = 1
            else:
                step = int(indices_split[2])

            for idx in range(start, stop + 1, step):
                raw_file_paths.append(objectfilename % idx)
                time_steps.append(idx)

            if len(raw_file_paths) == 0:
                raise Exception(
                    'Error in load_dat_raw: ObjectIndices found in file "' + dat_file_path
                    + '" lead to empty set of file names.')
        else:
            raw_file_paths.append(objectfilename)
        is_absolute_path = get_is_absolute_path(raw_file_paths[0])
        if not is_absolute_path:
            raw_file_paths = [get_path_to_file(dat_file_path) + raw_file_path for raw_file_path in raw_file_paths]

    if 'resolution' not in dat_dict:
        raise Exception('Error in load_dat_raw: Entry \'Resolution\' missing in "' + dat_file_path + '".')
    resolution_split = dat_dict['resolution'].split()
    if len(resolution_split) != 3:
        raise Exception(
            'Error in load_dat_raw: Entry \'Resolution\' in "' + dat_file_path + '" does not have three values.')
    xs = int(resolution_split[0])
    ys = int(resolution_split[1])
    zs = int(resolution_split[2])

    if 'format' not in dat_dict:
        raise Exception('Error in load_dat_raw: Entry \'Format\' missing in "' + dat_file_path + '".')
    format_string = dat_dict['format'].lower()
    if format_string == "float":
        num_components = 1
        bytes_per_entry = 4
    elif format_string == "uchar" or format_string == "byte":
        num_components = 1
        bytes_per_entry = 1
    elif format_string == "ushort":
        num_components = 1
        bytes_per_entry = 2
    elif format_string == "float3":
        num_components = 3
        bytes_per_entry = 4
    elif format_string == "float4":
        num_components = 4
        bytes_per_entry = 4
    else:
        raise Exception(
            'Error in load_dat_raw: Unsupported format \'' + format_string + '\' in file "' + dat_file_path + '".')

    # Load the .raw file(s) finally.
    raw_filename = raw_file_paths[len(raw_file_paths) - 1]
    with open(raw_filename, mode='rb') as file:
        if bytes_per_entry == 4:
            data = np.fromfile(file, dtype=np.float32)
        elif bytes_per_entry == 2:
            data = np.fromfile(file, dtype=np.uint16).astype(np.float32) / 65535.0
        elif bytes_per_entry == 1:
            data = np.fromfile(file, dtype=np.uint8).astype(np.float32) / 255.0
        array = np.reshape(data, [zs, ys, xs])

    return array


def load_nii(file_path, normalize=False):
    nii_file = nib.load(file_path)
    data_array = np.array(nii_file.get_fdata(), dtype=np.float32)
    if normalize:
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        data_array = (data_array - min_val) / (max_val - min_val)
    return data_array
