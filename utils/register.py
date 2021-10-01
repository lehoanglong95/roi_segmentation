import glob
import numpy as np
import pydicom
from skimage.util import montage
from skimage.transform import resize
import pylab as plt
import SimpleITK as sitk
# from dummy.preprocess import DSC


def load_dsc(dir_in=r'F:\Minh\projects\MRA\matlab_from_prof\BP429 2017-09-03\PWI', ):
    """
    Load DSC-MRP in 'dir_in' folder
    :param dir_in: the directory of the input image
    :return: preprocessed input, a brain mask and datasets (containing headers)
    """
    print('\n\nLoading DICOM files from %s\nPlease wait...' % dir_in)

    # dir_in = r'D:\workspace\BrainKUCMC\April2019\2019 CMC-contrast extra\Verio-Contrast-Normal\BP452 2018-10-04\PWI'

    series_filenames = glob.glob('%s/*.dcm' % dir_in)
    datasets = [pydicom.dcmread(filename) for filename in series_filenames]
    # pydicom.dcmread has a different set of fields than one obtain by dicomheader in MATLAB

    number_of_files = len(datasets)

    if 'siemens' in datasets[0].Manufacturer.lower():
        vendor = 'S'
        number_of_series = int(datasets[-1].AcquisitionNumber)
    else:
        vendor = 'G'
        number_of_series = int(datasets[0].NumberOfTemporalPositions)

    number_of_slice = int(number_of_files / number_of_series)

    # Check the pixel_spacing to determine whether interpolation is needed
    pixel_width, pixel_height = datasets[0].PixelSpacing[0], datasets[0].PixelSpacing[1]
    # If interpolation is needed, set upscale to a bi-linear transformation else to a Null function
    if pixel_width > 1:
        class Resize:
            def __init__(self, ratio_height, ratio_width):
                self.ratio = (ratio_height, ratio_width)

            def new_size_(self, x):
                return int(np.ceil(x.shape[-2] * self.ratio[0])), int(np.ceil(x.shape[-2] * self.ratio[1]))

            def resize_(self, x):
                return resize(x, self.new_size_(x))

        upscale = Resize(pixel_height, pixel_width).resize_
        new_size = Resize(pixel_height, pixel_width).new_size_(datasets[0].pixel_array)
        img = np.zeros((number_of_series, number_of_slice) + new_size)
    else:
        def foo(x):
            return x

        upscale = foo
        img = np.zeros((number_of_series, number_of_slice) + datasets[0].pixel_array.shape)

    if vendor == 'S':
        for i, ds in enumerate(datasets):
            InstanceNumber = int(ds.InstanceNumber - ((ds.AcquisitionNumber - 1) * number_of_slice) - 1)
            img[ds.AcquisitionNumber - 1, InstanceNumber] = upscale(ds.pixel_array)
    else:
        for i, ds in enumerate(datasets):
            InstanceNumber = int(number_of_slice - (
                    np.ceil((number_of_files - ds.InstanceNumber + 1.0) / number_of_series) - 1) - 1)
            img[ds.TemporalPositionIdentifier - 1, InstanceNumber] = upscale(ds.pixel_array)

    # Mask generation & input normalization
    img, mask = DSC().preprocess_raw_input(img)
    # print(datasets[0].pixel_array.shape, number_of_slice)
    print(img.shape)
    datasets[0].new_height = img.shape[-2]
    datasets[0].new_width = img.shape[-1]
    hdr = [datasets[i] for i in range(img.shape[1])]

    return img, mask, hdr


class Resize:
    def __init__(self, ratio_height, ratio_width):
        self.ratio = (ratio_height, ratio_width)
        self._new_size = None

    def new_size_(self, x, fix_size=(None, None)):
        new_h = int(np.ceil(x.shape[-2] * self.ratio[0])) if fix_size[0] is None else fix_size[0]
        new_w = int(np.ceil(x.shape[-1] * self.ratio[1])) if fix_size[1] is None else fix_size[1]
        self._new_size = new_h, new_w
        return new_h, new_w

    def resize_(self, x):
        return resize(x, self._new_size)


def through_plane_interpolate(target, ref):
    """
    Interpolation to match slice thickness of target and reference object
    :param target: ADC or DWI object
    :param ref: PWI object
    :return: ADC or DWI object
    """
    ratio = target.slice_thickness / ref.slice_thickness
    if ratio == 1:
        return target

    rz = Resize(ratio, ratio)
    upscale = rz.resize_
    shape = target.scan.shape
    new_size = rz.new_size_(target.scan[:, 0, :], fix_size=(None, shape[-1]))
    img = np.zeros((new_size[0], shape[-2], new_size[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[-2]):
            img[:, j, :] = upscale(target.scan[:, j, :])
    target.scan = img
    target.slice_thickness = ref.slice_thickness
    return target


def extract_itkimage(_itkimage):
    """Return numpy arrays of image, origin and spacing"""
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    scan = sitk.GetArrayFromImage(_itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(_itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(_itkimage.GetSpacing())))

    return scan, origin, spacing


def load_itk(_filename, itkimage_only=False):
    """This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image"""
    _itkimage = sitk.ReadImage(_filename)
    if itkimage_only:
        return _itkimage

    scan, origin, spacing = extract_itkimage(_itkimage)
    return scan, origin, spacing, _itkimage


def load_and_resample_itk(_filename, _ref_img, is_label=False, force_size=False):
    """Load and resample itkimage to _ref_img
    _filename: string or list of string or an sitk image
    """
    if isinstance(_filename, str) or isinstance(_filename, list):
        ori_itkimage = load_itk(_filename, itkimage_only=True) if isinstance(_filename, str) else _filename
    else:
        ori_itkimage = _filename

    # Compute size of resampled image
    ref_spacing = _ref_img.GetSpacing()
    ref_spacing = np.array([min(1, ref_spacing[0]), min(1, ref_spacing[1]),
                            ref_spacing[-1]])  # force the values of pixel-width, pixel-height to 1
    orig_size = np.array(ori_itkimage.GetSize(), dtype=np.int)
    orig_spacing = np.array(ori_itkimage.GetSpacing())
    if not force_size:
        new_size = orig_size * (orig_spacing / ref_spacing)
        new_size = list(np.round(new_size).astype(np.int))  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]
    else:
        new_size = _ref_img.GetSize()
    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    # We need both direction and origin to correctly resample the original image
    resampler.SetOutputDirection(_ref_img.GetDirection())  # direction from original image
    resampler.SetOutputOrigin(_ref_img.GetOrigin())  # origin from original image
    resampler.SetOutputSpacing(ref_spacing)
    resampler.SetSize(new_size)

    # Perform Resampling
    resampled_itkimage = resampler.Execute(ori_itkimage)
    return resampled_itkimage


class MyDicom:
    """The base class for other dicom class"""

    def __init__(self, dir_patient, image_type):
        self.dir_patient = dir_patient
        self.filenames = [filename for filename in glob.glob(f'{self.dir_patient}/{image_type}/*.dcm')]
        if not(len(self.filenames)):
            self.filenames = [filename for filename in glob.glob(f'{self.dir_patient}/*.dcm')]
        if not(len(self.filenames)):
            print('DICOM files not found')
            self.scan = None
            return
        if image_type == "ADC":
            del_temp = []
            for filename in self.filenames:
                temp = pydicom.dcmread(filename)
                if "ADC" not in temp.ImageType:
                    del_temp.append(filename)
            for filename in del_temp:
                self.filenames.remove(filename)
        else:
            del_temp = []
            for filename in self.filenames:
                temp = pydicom.dcmread(filename)
                if "ADC" in temp.ImageType:
                    del_temp.append(filename)
            for filename in del_temp:
                self.filenames.remove(filename)
        first_hdr = pydicom.dcmread(self.filenames[0])
        last_hdr = pydicom.dcmread(self.filenames[-1])
        if 'siemens' in first_hdr.Manufacturer.lower():
            self.vendor = 'S'
            self.slice_thickness = float(first_hdr.SliceThickness)
            num_temp_position = 1 if not hasattr(first_hdr, 'NumberOfTemporalPositions') else first_hdr.NumberOfTemporalPositions
            if (image_type.lower() == 'pwi') or (float(num_temp_position) > 1):
                self.number_of_slice = int(len(self.filenames) / int(last_hdr.AcquisitionNumber))
            else:
                self.number_of_slice = len(self.filenames)
        else:
            self.vendor = 'G'
            # self.slice_thickness = max(float(first_hdr.SliceThickness), float(first_hdr.SpacingBetweenSlices))
            try:
                self.slice_thickness = float(first_hdr.SliceThickness)
            except:
                self.slice_thickness = float(first_hdr.PixelMeasuresSequence[0].SliceThickness)
            if (image_type.lower() == 'pwi') or (float(first_hdr.NumberOfTemporalPositions) > 1):
                self.number_of_slice = int(len(self.filenames) / int(first_hdr.NumberOfTemporalPositions))
            else:
                self.number_of_slice = len(self.filenames)
        self.first_hdr = first_hdr
        self.image_type = image_type
        files = self.filenames[0] if image_type.lower() == 'pwi' else self.filenames
        files = sort_files(files)
        self.scan, self.origin, self.spacing, self.itkimage = load_itk(files)

    @staticmethod
    def remove_empty_slices(target):
        """
        Remove empty slices appeared after the resampling process
        :param target: ADC or DWI object
        :return: ADC or DWI object
        """
        target.scan = target.scan[np.where(target.scan.sum(axis=(1, 2)) > 0)]
        return target


def sort_files(file_list):
    """Sort the list of dicom files in order before actually loading the dicom images"""
    file_reader = sitk.ImageFileReader()
    file_and_pos = []
    for f in file_list:
        file_reader.SetFileName(f)
        file_reader.ReadImageInformation()
        # file_and_pos.append((f, float(file_reader.GetMetaData('0020|1041').split('\\')[-1])))
        file_and_pos.append((f, file_reader.GetOrigin()[-1]))
    file_and_pos.sort(key=lambda x: x[1])
    sorted_file_list, _ = zip(*file_and_pos)
    return sorted_file_list


class Resampled:
    """"""

    def __init__(self, target, ref, is_label=False):
        """
        The class of  resampled images
        :param image_type: str
        :param target:
        :param ref:
        """
        self.__dict__.update(target.__dict__)
        self.itkimage = load_and_resample_itk(target.itkimage, ref.itkimage, is_label=is_label)
        self.scan, self.origin, self.spacing = extract_itkimage(self.itkimage)


class PWI(MyDicom):
    """PWI is used as a reference for the registration of ADC/DWI to the phase-map space. Only the first slice of PWI is loaded.
    Please use the function load_dsc for fully load the PWI images
    """

    def __init__(self, dir_patient):
        super(PWI, self).__init__(dir_patient, 'PWI')

    def resample(self, ref_img):
        resampled = Resampled(self, ref_img)
        return resampled


class ADC(MyDicom):
    """"""

    def __init__(self, dir_patient):
        super(ADC, self).__init__(dir_patient, 'ADC')

    def resample(self, ref_img):
        resampled = Resampled(self, ref_img)
        # Map the size of the image to the target image
        # resampled.scan = resampled.scan[:ref_img.number_of_slice, :ref_img.scan.shape[-2], :ref_img.scan.shape[-1]]
        return resampled


class GroundTruth:
    """"""
    def __init__(self, image, src_sitk=None, file_path=None):
        self.file_path = file_path
        self.itkimage = sitk.GetImageFromArray(image)
        if src_sitk is not None:
            self.itkimage.CopyInformation(src_sitk)

    def resample(self, ref_img):
        resampled = Resampled(self, ref_img, is_label=True)
        return resampled


class DWI(MyDicom):
    """"""

    def __init__(self, dir_patient):
        super(DWI, self).__init__(dir_patient, 'DWI')

    def resample(self, ref_img):
        """
        Match the spacing of the image with the ref_img
        :param ref_img: PWI object
        :return: ADC or DWI object
        """
        resampled = Resampled(self, ref_img)
        # Map the size of the image to the target image
        # resampled.scan = resampled.scan[:ref_img.number_of_slice, :ref_img.scan.shape[-2], :ref_img.scan.shape[-1]]
        return resampled


if __name__ == "__main__":
    dir_moving = '/home/minh/PycharmProjects/Segmentation_GUI/dummy/data/moving'
    dir_target = '/home/minh/PycharmProjects/Segmentation_GUI/dummy/data/target'

    target = ADC(dir_target)
    moving = ADC(dir_moving)
    gt = moving.copy()
    gt.scan = np.load('')
    moving_resampled = moving.resample(target)
    gt_resampled = gt.resample(target)

    plt.imshow(montage(target.scan), cmap='gray')
    plt.imshow(montage(moving_resampled.scan), cmap='jet', alpha=.2)

    # plt.imshow(montage(target.scan), cmap='gray')
    # plt.imshow(montage(moving.scan), cmap='gray', alpha=1)
    # plt.imshow(montage(moving_resampled.scan), cmap='jet', alpha=.2)

    # plt.contour(montage(moving_resampled.scan > target.scan.max() * .02), cmap='gray', alpha=.5)

    images_dict = {
        'ADC': moving,
        'ADC-resampled': moving_resampled,
        'ADC-target': target,
    }

    for k, img in images_dict.items():
        print('%s: ' % k,
              ' ( Shape', img.scan.shape, ')',
              ' ( Origin', img.origin, ')',
              ' ( Spacing', img.spacing, ')',
              ' ( Slice Thickness', img.slice_thickness, ')',
              )

    plt.show()
