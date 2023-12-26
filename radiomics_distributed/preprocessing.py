import concurrent.futures as cf
import datetime
import logging
import os

# import radiomics
import SimpleITK as sitk
import numpy as np
from radiomics_distributed import utils as ut


def bias_correction(input_itk_image, max_iteration_list, shrink_size=None):
    if shrink_size is None:
        shrink_size = [2,2,2]
    pre_bias_correction_image = sitk.Shrink(input_itk_image, shrink_size)
    # construct mask image to exclude near zero pixels
    maskImage = sitk.OtsuThreshold(input_itk_image, 0, 1, 200)
    maskImage = sitk.Shrink(maskImage, shrink_size)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(max_iteration_list)
    post_bias_correction_image = corrector.Execute(pre_bias_correction_image, maskImage)

    # acquire and resample bias field
    bias_field = (post_bias_correction_image + 0.01) / (pre_bias_correction_image + 0.01)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(input_itk_image.GetOrigin())
    resampler.SetOutputSpacing(input_itk_image.GetSpacing())
    resampler.SetSize(input_itk_image.GetSize())
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetOutputDirection(input_itk_image.GetDirection())
    bias_field = resampler.Execute(bias_field)
    input_itk_image = sitk.Cast(input_itk_image, sitk.sitkFloat32)
    return bias_field * (input_itk_image + 0.01) - 0.01

def preprocess_image(patient_folder_path, preprocessing_parameters, image_modality, image=None):
    """
    :param image_sitk: original image in SimpleITK format
    :param preprocessing_parameters: dictionary that specifies the preprocessing parameters
    :return: the preprocessed image without any resampling
    """
    if image is None:
        image = os.path.join(patient_folder_path, '{}.nii.gz'.format(image_modality))
    if isinstance(image, str) and os.path.exists(image):
        image_sitk = sitk.ReadImage(image)
    elif isinstance(image, sitk.Image):
        image_sitk = image
    else:
        logging.warning('Cannot acquire the input original image {0}.'.format(image))
        return
    image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
    # Bias correction
    bias_correction_parameters = preprocessing_parameters.get('biasCorrectionMaxIterations')
    if bias_correction_parameters is not None:
        logging.info('Performing bias correction...')
        image_sitk = bias_correction(image_sitk, bias_correction_parameters)
        # Empty slice removal
        image_array = sitk.GetArrayFromImage(image_sitk)
        origin = image_sitk.GetOrigin()
        spacing = image_sitk.GetSpacing()
        corrected_image_array = []
        min_pixel_value = np.min(image_array)
        logging.info('Performing empty slice removal...')
        for i in range(image_array.shape[0]):
            image_slice = image_array[i,:,:]
            if np.mean(image_slice) > min_pixel_value+abs(min_pixel_value*0.01):
                corrected_image_array.append(image_slice)
        if len(corrected_image_array) < image_array.shape[0]:
            logging.warning('Empty slice detected and corrected.')
        corrected_image_array = np.array(corrected_image_array)
        image_sitk = sitk.GetImageFromArray(corrected_image_array)
        image_sitk.SetOrigin(origin)
        image_sitk.SetSpacing(spacing)

    # normalization
    normalization_parameters = preprocessing_parameters.get('normalization')
    mean = None
    standard_deviation = None
    if isinstance(normalization_parameters, dict):
        reference_structure = normalization_parameters.get('referenceStructure')
        reference_mask_filepath = None
        if isinstance(reference_structure, str):
            reference_mask_filepath = os.path.join(patient_folder_path, reference_structure + '_total_mask.nii.gz')
        if reference_structure is not None and os.path.exists(reference_mask_filepath):
            logging.info('Performing normalization based on reference structure {0}.'.format(reference_structure))
            reference_mask = sitk.ReadImage(reference_mask_filepath)
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputOrigin(reference_mask.GetOrigin())
            resampler.SetSize(reference_mask.GetSize())
            resampler.SetOutputSpacing(reference_mask.GetSpacing())
            ## registration needed
            image_registration_path = os.path.join(patient_folder_path, image_modality+'_registration_matrix.json')
            if os.path.exists(image_registration_path):
                registration_matrix = ut.import_from_json(image_registration_path)
                registration_matrix_np = np.array(registration_matrix)
                final_transformation = sitk.AffineTransform(registration_matrix_np[0:-1, 0:-1].flatten().tolist(),
                    registration_matrix_np[0:-1, -1].tolist())
                final_transformation = final_transformation.GetInverse()
                resampler.SetTransform(final_transformation)
            ## registration needed
            resampled_image = resampler.Execute(image_sitk)

            resampled_image_array = sitk.GetArrayFromImage(resampled_image)
            mask = sitk.GetArrayFromImage(reference_mask).astype(bool)
            BS = resampled_image_array[mask]
            masked_mean = np.mean(BS)
            masked_std = np.std(BS)
            logging.info('The mean and standard deviation of the reference structure is {0}/{1} before normalization.'.format(masked_mean, masked_std))
            image_sitk = (image_sitk-masked_mean)/masked_std*normalization_parameters.get('standard deviation', 1)+normalization_parameters.get('mean', 0)
            BS = sitk.GetArrayFromImage(resampler.Execute(image_sitk))[mask]
            masked_mean = np.mean(BS)
            masked_std = np.std(BS)
            logging.info(
                'The mean and standard deviation of the reference structure is {0}/{1} after normalization.'.format(
                    masked_mean, masked_std))
        else:
            logging.info('Performing global normalization...')
            mean = normalization_parameters.get('mean', 0)
            standard_deviation = normalization_parameters.get('standardDeviation', 1)
            normalize_filter = sitk.NormalizeImageFilter()
            image_sitk = normalize_filter.Execute(image_sitk)*standard_deviation+mean

    thresholding_range = preprocessing_parameters.get('thresholding')
    if thresholding_range is not None:
        statistics_image_filter = sitk.StatisticsImageFilter()
        statistics_image_filter.Execute(image_sitk)
        if standard_deviation is None:
            standard_deviation = statistics_image_filter.GetVariance()**0.5
        if mean is None:
            mean = statistics_image_filter.GetMean()
        upper_threshold = mean+thresholding_range*standard_deviation
        lower_threshold = mean-thresholding_range*standard_deviation
        logging.info('Performing thresholding with maximum and minimum threshold values of {0}/{1}...'.format(upper_threshold, lower_threshold))
        threshold_filter = sitk.ThresholdImageFilter()
        threshold_filter.SetMaximum(upper_threshold)
        threshold_filter.SetMinimum(lower_threshold)
        image_sitk = threshold_filter.Execute(image_sitk)

    # ... more preprocessing steps can be added if necessary
    return image_sitk

def image_resegmentation(image_sitk, resegmentation_range):
    # resegmentation
    if resegmentation_range is None or image_sitk is None:
        return
    logging.info('Resegmenting image based on absolute thresholds {0}.'.format(resegmentation_range))
    binary_thresholding_filter = sitk.BinaryThresholdImageFilter()
    binary_thresholding_filter.SetLowerThreshold(resegmentation_range[0])
    binary_thresholding_filter.SetUpperThreshold(resegmentation_range[1])
    resegmentation_mask = binary_thresholding_filter.Execute(image_sitk)
    return resegmentation_mask


def bounding_box_from_roi_contour(roi_contour, margin=8):
    """
    Finding the bounding box coordinates from the roi contour coordinates.
    :param roi_contour: The contour point coordinates of a ROI in list
    :param margin: the margin size for the ROI mask
    :return: 2-level nested list: [[x_min, x_max],[y_min,y_max],[z_min, z_max]]
    """
    # input check
    #
    # z

    z_loc_list = []
    y_loc_list = []
    x_loc_list = []
    for slice in roi_contour:
        z_loc_list.append(slice[0][2])
        for xyz in slice:
            x_loc_list.append(xyz[0])
            y_loc_list.append(xyz[1])
    z_loc_list = np.array(z_loc_list)
    z_out = [z_loc_list.min()-margin, z_loc_list.max()+margin]
    y_loc_list = np.array(y_loc_list)
    y_out = [y_loc_list.min()-margin, y_loc_list.max()+margin]
    x_loc_list = np.array(x_loc_list)
    x_out = [x_loc_list.min()-margin, x_loc_list.max()+margin]
    return [x_out, y_out, z_out]


def single_patient_image_preprocessing(cleaned_patient_data_path, export_directory, preprocessing_parameters, roi_names):
    """
    Performing image preprocessing and image resampling based on the mask bounding boxes for each patient. Each image
    will be resampled to multiple versions corresponding to the specified ROI masks. The final preprocessed and
    resampled image and mask will be put to the same folder
    :param cleaned_patient_data_path: the path of the lowest level of cleaned patient data containing the image and masks
    :param preprocessing_parameters: dictionary that specifies the preprocessing parameters
    :param roi_names: names of the ROIs
    :return the minimum pixel value of the complete preprocessed image. The default pixel value of the resampled image
    is min_pixel_value-1. It is used to set the lower resegmentation threshold in feature extraction in case of the ROI
    exceeds the image region
    """
    # for roi_name in roi_names:
    #     roi_export_directory = os.path.join(export_directory, roi_name)
    #     if not os.path.exists(roi_export_directory):
    #         break
    #     print('Cleaned database {0} has been preprocessed.'.format(cleaned_patient_data_path))
    #     return
    preprocessed_images = dict()
    modality_specific_preprocessing_parameters = preprocessing_parameters.get('imageModalities', dict())
    for image_modality, single_modality_processing_parameters in modality_specific_preprocessing_parameters.items():
        logging.info('Preprocessing image {0} on {1}.'.format(image_modality, cleaned_patient_data_path))
        preprocessed_image = preprocess_image(cleaned_patient_data_path, single_modality_processing_parameters, image_modality)
        if preprocessed_image is None:
            print('Warning: cannot preprocess image for modality {0}.'.format(image_modality))
            continue
        preprocessed_images[image_modality] = preprocessed_image
        logging.info('Image preprocessing finished.')

    resampling_resolution = preprocessing_parameters.get('resampleResolution', [1, 1, 1])
    mask_resampler = sitk.ResampleImageFilter()
    mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_resampler.SetOutputSpacing(resampling_resolution)
    mask_resampler.SetDefaultPixelValue(0)
    mask_resampler.SetOutputPixelType(sitk.sitkUInt8)


    for roi_name in roi_names:
        roi_mask_filepath = os.path.join(cleaned_patient_data_path, roi_name + '_total_mask.nii.gz')

        if not os.path.exists(roi_mask_filepath):
            continue
        roi_export_directory = os.path.join(export_directory, roi_name)
        if not os.path.exists(roi_export_directory):
            os.makedirs(roi_export_directory)

        roi_mask = sitk.ReadImage(roi_mask_filepath)
        roi_bounding_box = np.array(ut.mask_to_bounding_box(roi_mask, margin_size=16))
        roi_origin = roi_bounding_box[:, 0]
        roi_size = np.ceil(
            (roi_bounding_box[:, 1] - roi_bounding_box[:, 0]) / np.array(resampling_resolution)).flatten().astype(
            'int').tolist()
        mask_resampler.SetOutputOrigin(roi_origin)
        mask_resampler.SetSize(roi_size)
        #mask_resampler.SetOutputDirection(roi_mask.GetDirection())
        resampled_mask = mask_resampler.Execute(roi_mask)
        sitk.WriteImage(resampled_mask, os.path.join(roi_export_directory, 'mask.mha'))
        # resegmentation_masks = dict()
        base_image_modality_keyword = 'Base image'
        base_image_modality = None
        for image_modality in preprocessed_images.keys():
            if base_image_modality_keyword in image_modality:
                base_image_modality = image_modality
                break
        base_image = preprocessed_images.get(base_image_modality)
        for image_modality, preprocessed_image in preprocessed_images.items():
            logging.info('Resampling on {0}, {1}.'.format(image_modality, roi_name))
            preprocessed_image = preprocessed_images[image_modality]
            # if image_modality != base_image_modality:
            #     preprocessed_image = image_registration(base_image,preprocessed_image, roi_export_directory, fixed_mask=resampled_mask)
                # sitk.WriteImage(preprocessed_image, os.path.join(roi_export_directory,image_modality + '_registered.mha'))
            # save image before resampling
            #sitk.WriteImage(preprocessed_image, os.path.join(roi_export_directory, image_modality + '_preprocessed.mha'))

            min_max_filter = sitk.MinimumMaximumImageFilter()
            min_max_filter.Execute(preprocessed_image)
            min_pixel_value = min_max_filter.GetMinimum()
            max_pixel_value = min_max_filter.GetMaximum()
            image_resampler = sitk.ResampleImageFilter()
            image_resampler.SetInterpolator(sitk.sitkBSpline)
            image_resampler.SetOutputSpacing(resampling_resolution)
            image_resampler.SetDefaultPixelValue(min_pixel_value - 10)
            image_resampler.SetOutputDirection(resampled_mask.GetDirection())

            '''
            image_registration_path = os.path.join(cleaned_patient_data_path, image_modality + '_registration_matrix.json')
            if os.path.exists(image_registration_path):
                registration_matrix = ut.import_from_json(image_registration_path)
                registration_matrix_np = np.array(registration_matrix)
                final_transformation = sitk.AffineTransform(registration_matrix_np[0:-1, 0:-1].flatten().tolist(),
                                                            registration_matrix_np[0:-1, -1].tolist())
                final_transformation = final_transformation.GetInverse()
                image_resampler.SetTransform(final_transformation)
            '''
            image_resampler.SetOutputOrigin(roi_origin)
            image_resampler.SetSize(roi_size)
            resampled_image = image_resampler.Execute(preprocessed_image)

            #sitk.WriteImage(resampled_image, os.path.join(roi_export_directory, image_modality + '_resampled.mha'))

            resegmentation_range = modality_specific_preprocessing_parameters.get(image_modality, dict()).get('resegmentationRange')
            if resegmentation_range is not None:
                resegmentation_range = [max(resegmentation_range[0], min_pixel_value), min(resegmentation_range[1], max_pixel_value+1)]
                resegmentation_mask = image_resegmentation(resampled_image, resegmentation_range)
                and_filter = sitk.AndImageFilter()
                resegmentation_mask = and_filter.Execute(resegmentation_mask, resampled_mask)
                sitk.WriteImage(sitk.Cast(resegmentation_mask, sitk.sitkUInt8),
                    os.path.join(roi_export_directory, image_modality + '_resegmented_mask.mha'))
            # else:
            #     resegmentation_range = [min_pixel_value, max_pixel_value]

            # resegmentation_masks[image_modality] = resegmentation_mask
            # resegmented_mask_array = sitk.GetArrayFromImage(resegmentation_mask).astype(bool)
            # resampled_image_array = sitk.GetArrayFromImage(resampled_image)
            # maximum_masked_value = np.max(resampled_image_array[resegmented_mask_array])
            # minimum_masked_value = np.min(resampled_image_array[resegmented_mask_array])
            # logging.info('The maximum/minimum value of the resegmented image is {0}/{1}.'.format(maximum_masked_value,
            #                                                                                      minimum_masked_value))
            sitk.WriteImage(resampled_image, os.path.join(roi_export_directory, image_modality+'.mha'))
        # for image_modality, resegmentation_mask in resegmentation_masks.items():
        #     resegmentation_modality = preprocessing_parameters.get(image_modality, dict()).get('resegmentation', dict()).get('modality')
        #     and_filter = sitk.AndImageFilter()
        #     if resegmentation_modality in resegmentation_masks:
        #         resegmentation_mask = and_filter.Execute(resegmentation_masks[resegmentation_modality],resegmentation_mask)
        #     resegmentation_mask = and_filter.Execute(resegmentation_mask, resampled_mask)
        #     sitk.WriteImage(sitk.Cast(resegmentation_mask, sitk.sitkUInt8), os.path.join(roi_export_directory,
        #                                                                                  image_modality+'_resegmented_mask.mha'))

def image_preprocessing(cleaned_data_path, export_directory, preprocessing_parameters):
    patient_ids = os.listdir(cleaned_data_path)
    batch_size = os.cpu_count()-2
    batch_number = 0
    batch_starting_indices = np.arange(0, len(patient_ids), batch_size)
    for batch_starting_index in batch_starting_indices:
        batch_number += 1
        futures = dict()
        with cf.ProcessPoolExecutor(max_workers=batch_size) as executor:
            for patient_index in range(batch_starting_index, batch_starting_index+batch_size):
                if patient_index >= len(patient_ids):
                    break

                patient_id = patient_ids[patient_index]
                # if patient_id != 'QEH_NPC_1_78':
                #     continue
                cleaned_patient_data_path = os.path.join(cleaned_data_path, patient_id)
                if not os.path.isdir(cleaned_patient_data_path):
                    continue
                patient_export_directory = os.path.join(export_directory, patient_id)
                if not os.path.exists(patient_export_directory):
                    os.makedirs(patient_export_directory)
                print('Image preprocessing job submitted for patient {0}.'.format(patient_id))
                future = executor.submit(single_patient_image_preprocessing, cleaned_patient_data_path, patient_export_directory,
                                                                     preprocessing_parameters)
                futures[future] = patient_id
            print('Preprocessing execution for patient batch {0}/{1}'.format(batch_number, len(batch_starting_indices)))
            for future in cf.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error('Error occured during image preprocessing on patient {0}: {1}'.format(futures[future], e))

def image_preprocessing_sync(cleaned_data_path, export_directory, preprocessing_parameters, image_modalities, roi_names):
    patient_ids = os.listdir(cleaned_data_path)
    count = 0
    for patient_id in patient_ids:
        count += 1
        cleaned_patient_data_path = os.path.join(cleaned_data_path, patient_id)
        if not os.path.isdir(cleaned_patient_data_path):
            continue
        patient_export_directory = os.path.join(export_directory, patient_id)
        if not os.path.exists(patient_export_directory):
            os.makedirs(patient_export_directory)
        print('Image preprocessing for patient {0}.'.format(patient_id))
        # if patient_id !=  'QEH_NPC_1_5':
        #     continue
        single_patient_image_preprocessing(cleaned_patient_data_path,patient_export_directory,preprocessing_parameters, image_modalities, roi_names)
        if count >= 5:
            return


def image_partition(image, mask, export_directory, margin_size=8, image_chunk_size=30):
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    image_partition_position_all_dimensions = []
    for i in range(image_array.ndim):
        image_partition_position = np.arange(0, image_array.shape[i], image_chunk_size)
        image_partition_position_all_dimensions.append(image_partition_position)
    image_partition_positions = np.meshgrid(*image_partition_position_all_dimensions, indexing='ij')
    image_partition_positions = [item.flatten() for item in image_partition_positions]
    image_partition_positions = list(zip(*image_partition_positions))
    print('Partitioning images into {0} chunks...'.format(len(image_partition_positions)))
    image_filepaths = dict()
    mask_filepaths = dict()
    for i, image_partition_position in enumerate(image_partition_positions):
        mask_starting_indexes = np.array(image_partition_position)
        image_starting_indexes = mask_starting_indexes - margin_size
        mask_ending_indexes = mask_starting_indexes+image_chunk_size
        image_ending_indexes = mask_ending_indexes + margin_size
        image_chunk = image_array
        mask_chunk = mask_array
        image_chunk_origin = []
        mask_chunk_origin = []
        for dimension, image_starting_index, image_ending_index, mask_starting_index, mask_ending_index in zip(range(len(mask_starting_indexes)), image_starting_indexes,
                                                           image_ending_indexes, mask_starting_indexes, mask_ending_indexes):
            if image_ending_index >= image_array.shape[dimension]:
                image_ending_index = image_array.shape[dimension] - 1
            if image_starting_index < 0:
                image_starting_index = 0
            if mask_ending_index >= image_array.shape[dimension]:
                mask_ending_index = image_array.shape[dimension] - 1
            if mask_starting_index < 0:
                mask_starting_index = 0
            image_chunk_origin.append(image_starting_index)
            mask_chunk_origin.append(mask_starting_index)
            image_chunk = np.take(image_chunk, np.arange(image_starting_index, image_ending_index + 1), axis=dimension, mode='clip')
            mask_chunk = np.take(mask_chunk, np.arange(mask_starting_index, mask_ending_index + 1), axis=dimension, mode='clip')
        try:
            if np.max(mask_chunk) == 0:
                continue
        except Exception as e:
            logging.error('{}: '+str(e).format(datetime.datetime.now()))
            continue
        left_padding_widths = np.array(mask_chunk_origin)-np.array(image_chunk_origin)
        right_padding_widths = np.array(image_chunk.shape)-np.array(mask_chunk.shape)-left_padding_widths
        mask_chunk = np.pad(mask_chunk, ((left_padding_widths[0], right_padding_widths[0]),
                                         (left_padding_widths[1], right_padding_widths[1]),
                                         (left_padding_widths[2], right_padding_widths[2])), 'constant', constant_values=0)
        image_chunk = sitk.GetImageFromArray(image_chunk)
        mask_chunk = sitk.GetImageFromArray(mask_chunk)
        mask_chunk = sitk.Cast(mask_chunk, sitk.sitkUInt8)
        chunk_name = [str(item) for item in image_partition_position]
        image_filepath = os.path.join(export_directory, 'image_'+'_'.join(chunk_name)+'.mha')
        mask_filepath = os.path.join(export_directory, 'mask_' + '_'.join(chunk_name) + '.mha')
        image_filepaths[tuple(image_partition_position)] = image_filepath
        mask_filepaths[tuple(image_partition_position)] = mask_filepath
        sitk.WriteImage(image_chunk, image_filepath)
        sitk.WriteImage(mask_chunk, mask_filepath)
        logging.info('({0}/{1}) Image and mask chunk as position {2} has been exported.'.format(i+1, len(image_partition_positions),chunk_name))
    print('Image partition competed.')
    return image_filepaths, mask_filepaths

def image_concatenation(image_chunk_export_directory, original_image_path, kernel_size):
    original_image = sitk.GetArrayFromImage(sitk.ReadImage(original_image_path))
    original_image_size = original_image.shape
    del original_image
    feature_map_filenames = dict()
    for feature_map_filename in os.listdir(image_chunk_export_directory):
        if 'image' not in feature_map_filename:
            continue
        image_positions = os.path.splitext(feature_map_filename)[0].split('_')
        if len(image_positions) != 4:
            continue
        # print(image_positions)
        image_positions = [int(i) for i in image_positions[1:]]
        feature_map_filenames[tuple(image_positions)] = os.path.join(image_chunk_export_directory,feature_map_filename)
    print('{0} image chunks discovered. Starting image concatenation...'.format(len(feature_map_filenames)))
    image_positions = np.array(list(feature_map_filenames.keys()))
    coordinates = []
    for i in range(image_positions.shape[1]):
        sorted_image_positions = np.sort(np.unique(image_positions[:, i]))
        coordinates.append(sorted_image_positions)
    mesh_coordinates = np.meshgrid(*coordinates, indexing='ij')
    mesh_coordinates = [item.flatten() for item in mesh_coordinates]
    mesh_coordinates = list(zip(*mesh_coordinates))
    feature_map_combined = np.zeros(original_image_size)
    for starting_coordinate in mesh_coordinates:
        starting_coordinate = tuple(starting_coordinate)
        if starting_coordinate not in feature_map_filenames:
            continue
        feature_map_image = sitk.ReadImage(feature_map_filenames[starting_coordinate])
        feature_map_array = sitk.GetArrayFromImage(feature_map_image)
        start_index = []
        end_index = []
        feature_chunk = feature_map_array
        for d, index in enumerate(starting_coordinate):
            shape = feature_map_array.shape[d]
            starting_image_coordinate = index
            if index == 0:
                starting_chunk_index = 0
            else:
                starting_chunk_index = kernel_size
            ending_chunk_index = shape-1
            ending_image_coordinate = shape-starting_chunk_index+starting_image_coordinate-1
            if ending_image_coordinate >= original_image_size[d]:
                ending_image_coordinate = original_image_size[d]-1
                ending_chunk_index = starting_chunk_index-starting_image_coordinate+original_image_size[d]-1
            start_index.append(starting_image_coordinate)
            end_index.append(ending_image_coordinate)
            feature_chunk = np.take(feature_chunk,np.arange(starting_chunk_index, ending_chunk_index+1), axis=d)
        feature_map_combined[start_index[0]:end_index[0]+1,start_index[1]:end_index[1]+1,start_index[2]:end_index[2]+1] = feature_chunk
    feature_map_image = sitk.GetImageFromArray(feature_map_combined)
    print('Image concatenation completed.')
    return feature_map_image

