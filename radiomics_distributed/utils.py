import json
import os
from math import ceil

import SimpleITK as sitk
import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import RegularGridInterpolator, interp1d


def align_masks(masks):
    if len(masks) == 0:
        return masks
    origins = []
    absolute_sizes = []
    spacings = []
    for mask in masks:
        origins.append(mask.GetOrigin())
        absolute_sizes.append(np.array(mask.GetSize())*np.array(mask.GetSpacing()))
        spacings.append(mask.GetSpacing())
    origin = np.min(origins, axis=0).tolist()
    spacing = np.min(spacings, axis=0).tolist()
    size = np.ceil(np.max(absolute_sizes, axis=0)/spacing).astype(int).tolist()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(size)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    aligned_masks = [resampler.Execute(mask) for mask in masks]
    return aligned_masks

def crop_mask(mask_image:sitk.Image, margin_size=8, spacing = None, allow_free_rotation=False):
    bounding_box = mask_to_bounding_box(mask_image, margin_size=margin_size)
    if spacing is None:
        spacing = mask_image.GetSpacing()
    if allow_free_rotation:
        diagonal_distance = ((bounding_box[0][1]-bounding_box[0][0])**2+(bounding_box[1][1]-bounding_box[1][0])**2)**0.5
        box_length = diagonal_distance/np.sqrt(2)
        center = [int((position[1]+position[0])/2) for position in bounding_box]
        origin = [center[0]-box_length/2,center[1]-box_length/2,bounding_box[2][0]]
        size = [ceil(box_length/spacing[0]),ceil(box_length/spacing[1]),ceil((bounding_box[2][1]-bounding_box[2][0])/spacing[2])]
    else:
        origin = [position[0] for position in bounding_box]
        size = [ceil((position[1]-position[0])/spacing_1d) for position, spacing_1d in zip(bounding_box, spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(origin)
    resampler.SetSize(size)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    cropped_image = resampler.Execute(mask_image)
    return cropped_image


def mask_to_bounding_box(mask_image:sitk.Image, margin_size=8):
    origin = mask_image.GetOrigin()
    resolution = mask_image.GetSpacing()
    size = mask_image.GetSize()
    mask = sitk.GetArrayFromImage(mask_image)
    coordinates = []
    for i in range(3):
        coordinates.append(np.arange(size[i])*resolution[i]+origin[i])
    boundary_index = np.array(
        [[len(coordinates[2]) - 1, 0], [len(coordinates[1]) - 1, 0], [len(coordinates[0]) - 1, 0]])
    for i in range(len(coordinates)):
        for j in range(mask.shape[i]):
            slice_included = np.any(np.take(mask, j, axis=i))
            if slice_included:
                if j < boundary_index[i, 0]:
                    boundary_index[i, 0] = j
                if j > boundary_index[i, 1]:
                    boundary_index[i, 1] = j
    mask_boundary = [[coordinates[0][boundary_index[2, 0]], coordinates[0][boundary_index[2, 1]]],
                     [coordinates[1][boundary_index[1, 0]], coordinates[1][boundary_index[1, 1]]],
                     [coordinates[2][boundary_index[0, 0]], coordinates[2][boundary_index[0, 1]]]]
    for i in range(len(mask_boundary)):
        mask_boundary[i][0] = max(mask_boundary[i][0], coordinates[i][0])-margin_size
        mask_boundary[i][1] = min(mask_boundary[i][1], coordinates[i][-1])+margin_size
    return mask_boundary

def contour_to_mask(contour_sequence, coordinates):
    y_coordinates = coordinates[1]
    x_coordinates = coordinates[0]
    z_coordinates = coordinates[2]
    mask_3d = []
    slice_positions = []
    x_coordinate_interpolator = interp1d(x_coordinates, range(len(x_coordinates)), kind='nearest',
                                         bounds_error=False, fill_value=(0, len(x_coordinates) - 1),
                                         assume_sorted=True)
    y_coordinate_interpolator = interp1d(y_coordinates, range(len(y_coordinates)), kind='nearest',
                                         bounds_error=False, fill_value=(0, len(y_coordinates) - 1),
                                         assume_sorted=True)
    z_coordinate_interpolator = interp1d(z_coordinates, range(len(z_coordinates)), kind='nearest',
                                         bounds_error=False, fill_value=(0, len(z_coordinates) - 1),
                                         assume_sorted=True)
    for j in range(len(contour_sequence)):
        contour_points = np.array(contour_sequence[j], np.double)
        if contour_points.shape[0] < 2:
            continue
        contour_points = np.array(contour_sequence[j], np.double)
        contour_points = contour_points.reshape((-1, 3))
        contour_points_index = np.concatenate((x_coordinate_interpolator(contour_points[:, 0].reshape(-1, 1)),
                                               y_coordinate_interpolator(contour_points[:, 1].reshape(-1, 1)),
                                               z_coordinate_interpolator(contour_points[:, 2].reshape(-1, 1))),
                                              axis=1)

        contour_points_index = contour_points_index[:, 0:2].astype(int)
        contour_points_index = contour_points_index.reshape((-1, 1, 2))
        mask = np.zeros((len(y_coordinates),
                         len(x_coordinates),
                         1), np.uint8)
        mask = cv2.fillPoly(mask, [contour_points_index], 1)
        slice_position = contour_points[0, 2]
        if slice_position in slice_positions:
            slice_index = slice_positions.index(slice_position)
            mask_3d[slice_index] = np.any([mask_3d[slice_index], mask[:, :, 0]], axis=0)
        else:
            slice_positions.append(slice_position)
            mask_3d.append(mask[:, :, 0])
    # mask_3d = mask_3d[:, :, 0:effective_dimension_z]
    ascending_index = np.argsort(slice_positions)
    slice_positions = np.array(slice_positions)[ascending_index]
    mask_3d = np.array(mask_3d).transpose((1,2,0))[:, :, ascending_index]
    mask_resampled_interpolator = RegularGridInterpolator((y_coordinates,
                                                           x_coordinates,
                                                           slice_positions),
                                                          mask_3d,
                                                          method='nearest',
                                                          bounds_error=False,
                                                          fill_value=0)
    x_mesh_coordinates, y_mesh_coordinates, z_mesh_coordinates = np.meshgrid(x_coordinates,
                                                                             y_coordinates,
                                                                             z_coordinates)
    flattened_mesh_points = np.transpose([y_mesh_coordinates.flatten(),
                                          x_mesh_coordinates.flatten(),
                                          z_mesh_coordinates.flatten()])
    resampled_mask = mask_resampled_interpolator(flattened_mesh_points)
    resampled_mask = resampled_mask > 0.5
    resampled_mask = resampled_mask.reshape(x_mesh_coordinates.shape)
    resampled_mask_sitk = sitk.Cast(sitk.GetImageFromArray(resampled_mask.transpose((2,0,1)).astype(int)), sitk.sitkUInt8)
    resampled_mask_sitk.SetOrigin([coordinates[0][0], coordinates[1][0], coordinates[2][0]])
    resampled_mask_sitk.SetSpacing([coordinates[0][1]-coordinates[0][0], coordinates[1][1]-coordinates[1][0],
                                    coordinates[2][1]-coordinates[2][0]])
    return resampled_mask_sitk


def mask_to_contour(mask_sitk=None, num_of_points_per_slice=None, mask_array=None, coordinates=None):
    if mask_sitk is not None:
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        coordinates = []
        for origin_1d, spacing_1d, size_1d in zip(mask_sitk.GetOrigin(), mask_sitk.GetSpacing(), mask_sitk.GetSize()):
            coordinates.append(np.arange(size_1d) * spacing_1d + origin_1d)
    if mask_array is None or coordinates is None:
        return
    contours = []
    for i in range(mask_array.shape[0]):
        z_coordinate = coordinates[2][i]
        single_slice_combined_mask = mask_array[i, :, :]
        if np.sum(single_slice_combined_mask) == 0:
            continue
        # single_slice_combined_contour = ss.find_boundaries(single_slice_combined_mask)
        single_slice_combined_contour, hierarchy = cv2.findContours(single_slice_combined_mask.astype('uint8'),
                                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for single_slice_single_region_contour in single_slice_combined_contour:
            single_slice_single_region_contour = np.flip(single_slice_single_region_contour.flatten().reshape((-1, 2)), axis=1)
            single_slice_combined_contour_points = np.array([coordinates[0][single_slice_single_region_contour[:, 1] - 1],
                                                             coordinates[1][single_slice_single_region_contour[:, 0] - 1],
                                                             np.ones(single_slice_single_region_contour.shape[
                                                                         0]) * z_coordinate]).transpose()
            if num_of_points_per_slice is not None:
                single_slice_combined_contour_points = resample_contour(single_slice_combined_contour_points,
                                                                        num_of_points_per_slice)
            contours.append(single_slice_combined_contour_points.tolist())

        # single_slice_combined_contour = np.concatenate(
        #     [np.flip(item.flatten().reshape((-1, 2)), axis=1) for item in single_slice_combined_contour], 0)
        # single_slice_combined_contour_points = np.array([coordinates[0][single_slice_combined_contour[:, 1] - 1],
        #                                                  coordinates[1][single_slice_combined_contour[:, 0] - 1],
        #                                                  np.ones(single_slice_combined_contour.shape[
        #                                                              0]) * z_coordinate]).transpose()
        # single_slice_combined_contour[:,0] = common_coordinates[1][single_slice_combined_contour[:,0]-1]
        # single_slice_combined_contour[:, 1] = common_coordinates[0][single_slice_combined_contour[:, 1]-1]
        # single_slice_combined_contour=np.concatenate((single_slice_combined_contour,np.ones((single_slice_combined_contour.shape[0],1))*z_coordinate),axis=1)
        # single_slice_contour_mask = np.zeros(single_slice_combined_mask.shape)
        # single_slice_contour_mask[single_slice_combined_contour[:,0],single_slice_combined_contour[:,1]] = 1
        # if num_of_points_per_slice is not None:
        #     single_slice_combined_contour_points = resample_contour(single_slice_combined_contour_points,num_of_points_per_slice)
        # contours.append(single_slice_combined_contour_points.tolist())
    return contours, coordinates


def parse_from_yaml(filename):
    if not os.path.exists(filename):
        return
    with open(filename, 'r') as stream:
        try:
            result = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return
        return result

def export_from_dict_to_csv(export_file_name, export_dict):
    dataframe = pd.DataFrame({k: pd.Series(l) for k, l in export_dict.items()})
    # dataframe.fillna(0)
    dataframe.to_csv(export_file_name, index=False)
    return dataframe

def export_to_json(full_filename, instance):
    instance_json = json.dumps(instance, indent=4)
    export_file = open(full_filename, 'w')
    export_file.write(instance_json)
    export_file.close()


def import_from_json(full_filename):
    export_file = open(full_filename, 'r')
    instance_json = export_file.read()
    export_file.close()
    try:
        return json.loads(instance_json)
    except Exception as e:
        print('Error occured while importing {0}: {1}'.format(full_filename, e))
        return

def resample_contour(contour_list, num_of_points):
    contour_array = np.array(contour_list)
    if contour_array.shape[0] < num_of_points:
        return contour_list
    x = list(contour_array[:,0])
    y = list(contour_array[:,1])
    z = list(contour_array[:,2])
    # close the contour, temporarily
    if x[-1] != x[0]:
        x.append(x[0])
        y.append(y[0])
        z.append(z[0])

    dx = np.diff(x)
    dy = np.diff(y)

    dS = np.sqrt(np.square(dx) + np.square(dy))
    dS = np.concatenate(([0],dS))

    d = np.cumsum(dS)
    perim = d[-1]

    ds = perim / (num_of_points-1)
    dSi = ds * np.arange(num_of_points)

    dSi[-1] = dSi[-1] - 0.005

    xi = np.interp(dSi, d, x)
    yi = np.interp(dSi, d, y)
    zi = np.interp(dSi, d, z)

    return np.array([xi,yi,zi]).transpose()




########################################################################################################################
def resampleImg_to_size(img, outsize=None, resamplemethod=sitk.sitkLinear):
    ## resample SimpleITK image to a given size
    if outsize is None:
        outsize = [64, 64, 64]

    inputsize = img.GetSize()
    inputspacing = img.GetSpacing()

    outspacing = [inputsize[0] * inputspacing[0] / outsize[0],
                  inputsize[1] * inputspacing[1] / outsize[1],
                  inputsize[2] * inputspacing[2] / outsize[2]]

    transform = sitk.Transform()
    transform.SetIdentity()


    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(resamplemethod)  # sitk.sitkLinear / sitk.sitkNearestNeighbor
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetSize(outsize)
    newimg = resampler.Execute(img)

    return newimg


def resampleImg_to_resolution(img, outspacing=None, resamplemethod=sitk.sitkLinear):
    ## resample SimpleITK image to a given resolution
    if outspacing is None:
        outspacing = [1, 1, 1]

    inputsize = img.GetSize()
    inputspacing = img.GetSpacing()

    outsize = [int(inputsize[0] * inputspacing[0] / outspacing[0]),
               int(inputsize[1] * inputspacing[1] / outspacing[1]),
               int(inputsize[2] * inputspacing[2] / outspacing[2])]

    transform = sitk.Transform()
    transform.SetIdentity()

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(resamplemethod)  # sitk.sitkLinear / sitk.sitkNearestNeighbor
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetSize(outsize)
    newimg = resampler.Execute(img)

    return newimg


def resampleImg_as(input_img, ref_img, interpolator=sitk.sitkLinear):
    Image = input_img
    RefImage = ref_img

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetReferenceImage(RefImage)
    ResImage = resampler.Execute(Image)

    return ResImage