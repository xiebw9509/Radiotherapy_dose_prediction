# @paper202305
import tempfile
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
import datetime
import numpy as np
import SimpleITK as sitk
import os
import shutil
from configs import opt

def generate_rtdose(dose_volume_path, ct_dir, struct_path, spacing, save_dir, source_dir):
    struct_ds = pydicom.read_file(struct_path, force=True)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_dir)
    reader.SetFileNames(dicom_names)
    itk_ct = reader.Execute()
    origin = itk_ct.GetOrigin()
    orient = itk_ct.GetDirection()

    itk_doses = sitk.ReadImage(dose_volume_path)
    dose_volume = sitk.GetArrayFromImage(itk_doses)
    dose_volume = dose_volume / 100
    # # Create some temporary filenames
    # suffix = '.dcm'
    # filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name
    # filename_big_endian = tempfile.NamedTemporaryFile(suffix=suffix).name

    # Populate requried values for file meta information
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.RTDoseStorage
    file_meta.MediaStorageSOPInstanUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = '1.2.276.0.7230010.3.0.3.5.4'
    file_meta.ImplementationVersionName= 'OFFIS_DCMBP_354'
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    # Create the FileDataset instance
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient
    ds.PatientName = struct_ds.PatientName
    ds.PatientID = struct_ds.PatientID
    ds.PatientSex = struct_ds.PatientSex
    ds.PatientBirthDate = struct_ds.PatientBirthDate

    # set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Set creation data/time
    dt = datetime.datetime.now()
    dateStr = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')


    # General Study
    ds.StudyDate = dateStr
    ds.StudyTime = timeStr
    ds.StudyInstanceUID = struct_ds.StudyInstanceUID
    ds.StudyID = struct_ds.StudyID

    # RTSeries
    ds.Modality = 'RTDOSE'
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesNumber = 1

    # Frame of Reference
    ds.FrameOfReferenceUID = struct_ds.FrameOfReferenceUID

    # Gerneral Equipment
    ds.Manufacturer = ''
    ds.ManufacturerModelName = 'AI RTDose'
    ds.SeriesDescription = 'AI RTDose'
    # General Image
    ds.ContentDate = dateStr
    ds.ContentTime = timeStr
    ds.InstanceNumber = 1

    # Image Plane
    ds.ImagePositionPatient = r'{}\{}\{}'.format(*(origin))
    ds.ImageOrientationPatient = r'{}\{}\{}\{}\{}\{}'.format(*(orient))

    # Multi-Frame
    ds.FrameIncrementPointer = [(0x3004, 0x000C)]

    # RTDose
    ds.DoseGridScaling = dose_volume.max() / (2**16)
    ds.DoseSummationType = 'PLAN'
    ds.DoseType = 'PHYSICAL'
    ds.DoseUnits = 'GY'
    ds.GridFrameOffsetVector = generate_grid_frame_offset_vector(dose_volume.shape[0], spacing[2])

    dose_volume = (dose_volume / ds.DoseGridScaling)
    dose_volume = dose_volume.astype(np.int32)

    # SOP Common
    ds.InstancCreationDate = dateStr
    ds.InstancCreationTime = timeStr

    ds.SOPClassUID = pydicom._storage_sopclass_uids.RTDoseStorage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SpecificCharacterSet = 'ISO_IR 100'

    # Image Info
    ds.Rows = dose_volume.shape[1]
    ds.Columns = dose_volume.shape[2]
    ds.NumberOfFrames = dose_volume.shape[0]
    ds.BitsStored = 32
    ds.BitsAllocated = 32
    ds.SamplesPerPixel = 1
    ds.HighBit = 31
    ds.PixelSpacing = r'{}\{}'.format(spacing[0], spacing[1])
    ds.PixelRepresentation = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.PixelData = dose_volume[:].tobytes()

    # refer to structure
    ds.ReferencedStructureSetSequence = pydicom.Sequence()
    item = Dataset()
    ds.ReferencedStructureSetSequence.append(item)
    item.ReferencedSOPInstanceUID = struct_ds.SOPInstanceUID

    shutil.copytree(source_dir, save_dir)
    pred_dir = os.path.join(save_dir, 'Pred RTDOSE')
    os.makedirs(pred_dir, exist_ok=True)
    ds.save_as(os.path.join(pred_dir, 'pred_dose.dcm'), write_like_original=False)

def generate_grid_frame_offset_vector(slice_count, slice_thickness):

    for index in range(slice_count):
        if index == 0:
            res = r'{}'.format(index * slice_thickness)
        else:
            res += r'\{}'.format(index * slice_thickness)

    return res

if __name__ == '__main__':
    data_dir = '/Data/RemoteData'
    save_dir = 'predict_results'
    ct_dirs = []
    uids = []
    struct_paths = []
    dose_volume_paths = []
    dose_volume_dir = '/Data/DoseDistributionPrediction/CervicalCancerDoseDistributionPrediction/train/train2distance_map_l1_loss_resnet/predict/predict_388/result'

    test_split_file = os.path.join(opt.data_root, 'test_series.txt')

    with open(test_split_file, 'r') as f:
        indices = f.read().splitlines()

    for indice in indices:
        items = indice.split('/')
        uid = items[-1]
        print(uid)
        ct_dir = os.path.join(data_dir, indice[indice.find('/')+1:indice.rfind('/')])
        struct_dir = os.path.join(data_dir, indice[indice.find('/')+1:indice.find('CT')-1], 'RTSTRUCT')

        for root, dirs, files in os.walk(struct_dir):
            for file in files:
                if file == '{}.dcm'.format(uid):
                    struct_path = os.path.join(root, file)
                    break

        dose_volume_path = os.path.join(dose_volume_dir, '{}/volume/dose.nii.gz'.format(uid))
        save_sub_dir = os.path.join(save_dir, uid)
        source_dir = ct_dir[:ct_dir.find('CT')-1]
        generate_rtdose(dose_volume_path, ct_dir, struct_path, (2.5, 2.5, 2.5), save_sub_dir, source_dir)



