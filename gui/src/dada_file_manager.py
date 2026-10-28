import numpy as np
import os

class dada_file_manager(object):
    def __init__(self, file_name):
        self.file_abspath = os.path.abspath(file_name)
        self.file_nameonly = self.file_abspath.split("/")[-1]
        self.file_basename, self.file_extension = os.path.splitext(self.file_nameonly)
        self.dict_properties = self.read_DADA_header(self.file_abspath)

        self.header                   = self.dict_properties['HEADER']
        self.hdr_version              = float(self.dict_properties['HDR_VERSION'])
        self.hdr_size                 = int(self.dict_properties['HDR_SIZE'])
        self.dada_version             = float(self.dict_properties['DADA_VERSION'])
        self.pic_version              = float(self.dict_properties['PIC_VERSION'])
        self.obs_id                   = self.dict_properties['OBS_ID']
        self.primary                  = self.dict_properties['PRIMARY']
        self.secondary                = self.dict_properties['SECONDARY']
        self.file_name                = self.dict_properties['FILE_NAME']
        self.file_size                = int(self.dict_properties['FILE_SIZE'])
        self.file_number              = int(self.dict_properties['FILE_NUMBER'])
        self.utc_start                = self.dict_properties['UTC_START']
        self.mjd_start                = np.float64(self.dict_properties['MJD_START'])
        self.obs_offset               = int(self.dict_properties['OBS_OFFSET'])
        self.obs_overlap              = int(self.dict_properties['OBS_OVERLAP'])
        self.source_name              = self.dict_properties['SOURCE']
        self.ra                       = self.dict_properties['RA']
        self.dec                      = self.dict_properties['DEC']
        self.telescope                = self.dict_properties['TELESCOPE']
        self.instrument               = self.dict_properties['INSTRUMENT']
        self.freq_MHz                 = float(self.dict_properties['FREQ'])
        self.bw_MHz                   = float(self.dict_properties['BW'])
        self.t_samp_us                = float(self.dict_properties['TSAMP'])
        self.nbit                     = int(self.dict_properties['NBIT'])
        self.ndim                     = int(self.dict_properties['NDIM'])
        self.npol                     = int(self.dict_properties['NPOL'])
        self.nchan                    = int(self.dict_properties['NCHAN'])
        self.resolution               = int(self.dict_properties['RESOLUTION'])
        self.dsb                      = int(self.dict_properties['DSB'])

        self.value                    = None

    def read_DADA_header(self, filename):

        #dada_file = open(filename, "r") # valid for Python2.x
        #dada_file = open(filename, "r", encoding='utf-8') # valid for Python3.x
        # utf-8 raises errors with 0xda byte so ISO-8859-1 encoding should be used
        dada_file = open(filename, "r", encoding='ISO-8859-1') # valid for Python3.x
        dada_file_header = dada_file.read(4096).split("# end of header")[0]

        print(dada_file_header)

        list_header_lines = dada_file_header.split("\n")

        dict_properties = {}
        for line in list_header_lines:
            if line != "" and (not line.startswith("#")):
                key, value = line.split()[:2]
                #print(key.strip(), value.strip())
                dict_properties[key.strip()] = value.strip()

        #value(str(dict_properties['HEADER']))
        #print('Extracted value of dictionary ', dict_properties['HEADER'])
        #print('Extracted length of dictionary ', str(len(dict_properties)))

        dada_file.close()

        return dict_properties

    def write_DADA_header(self, outfilename, dict_params):

        outfile = open(outfilename, "w")

        outfile.write("HEADER       %s      # Distributed aquisition and data analysis\n" % dict_params['HEADER'])
        outfile.write("HDR_VERSION  %.1f    # Version of this ASCII header\n" % float(dict_params['HDR_VERSION']))
        outfile.write("HDR_SIZE     %s      # Size of the header in bytes\n" % dict_params['HDR_SIZE'])
        outfile.write("\n")
        outfile.write("DADA_VERSION %.1f    # Version of the DADA Software\n" % float(dict_params['DADA_VERSION']))
        outfile.write("PIC_VERSION  %.1f    # Version of the PIC FPGA Software\n" % float(dict_params['PIC_VERSION']))
        outfile.write("\n")
        outfile.write("# DADA parameters\n")
        outfile.write("OBS_ID       %s      # observation ID\n" % dict_params['OBS_ID'])
        outfile.write("PRIMARY      %s      # primary node host name\n" % dict_params['PRIMARY'])
        outfile.write("SECONDARY    %s      # secondary node host name\n" % dict_params['SECONDARY'])
        outfile.write("FILE_NAME    %s      # full path of the data file\n" % dict_params['FILE_NAME'])
        outfile.write("\n")
        outfile.write("FILE_SIZE    %d      # requested size of data files\n" % int(dict_params['FILE_SIZE']))
        outfile.write("FILE_NUMBER  %d      # number of data file\n" % int(dict_params['FILE_NUMBER']))
        outfile.write("\n")
        outfile.write("# time of the rising edge of the first time sample\n")
        outfile.write("UTC_START    %s      # yyyy-mm-dd-hh:mm:ss.fs\n" % dict_params['UTC_START'])
        outfile.write("MJD_START    %.29f   # MJD equivalent to the start UTC\n" % np.float64(dict_params['MJD_START']))
        outfile.write("\n")
        outfile.write("OBS_OFFSET   %d      # bytes offset from the start MJD/UTC\n" % int(dict_params['OBS_OFFSET']))
        outfile.write("OBS_OVERLAP  %d      # bytes by which neighbouring files overlap\n" % int(dict_params['OBS_OVERLAP']))
        outfile.write("\n")
        outfile.write("# description of the source\n")
        outfile.write("SOURCE       %s      # name of the astronomical source\n" % dict_params['SOURCE'])
        outfile.write("RA           %s      # Right Ascension of the source\n" % dict_params['RA'])
        outfile.write("DEC          %s      # Declination of the source\n" % dict_params['DEC'])
        outfile.write("\n")
        outfile.write("# description of the instrument\n")
        outfile.write("TELESCOPE    %s      # telescope name\n" % dict_params['TELESCOPE'])
        outfile.write("INSTRUMENT   %s      # instrument name\n" % dict_params['INSTRUMENT'])
        outfile.write("FREQ         %.3f    # centre frequency in MHz\n" % float(dict_params['FREQ']))
        outfile.write("BW           %.3f    # bandwidth of in MHz (-ve lower sb\n" % float(dict_params['BW']))
        outfile.write("TSAMP        %.6f    # sampling interval in microseconds\n" % float(dict_params['TSAMP']))
        outfile.write("\n")
        outfile.write("NBIT         %d      # number of bits per sample\n" % int(dict_params['NBIT']))
        outfile.write("NDIM         %d      # dimension of samples (2=complex, 1=real)\n" % int(dict_params['NDIM']))
        outfile.write("NPOL         %d      # number of polarizations observed\n" % int(dict_params['NPOL']))
        outfile.write("NCHAN        %d      # number of channels here\n" % int(dict_params['NCHAN']))
        outfile.write("RESOLUTION   %d      # a parameter that is unclear\n" % int(dict_params['RESOLUTION']))
        outfile.write("DSB          %d\n" % int(dict_params['DSB']))
        outfile.write("# end of header\n")
        outfile.close()

    def write_DADA_data(self, outfilename, data, data_length):

        # fill remaining bytes of header until 4096 is reached

        header_size_bytes = os.path.getsize(outfilename)
        #how many zeros have to be added after header till the 4096th byte
        null_bytes_to_write = 4096 - header_size_bytes

        #open the file in append/binary mode
        outfile = open(outfilename, 'ab')
        array_to_write = np.zeros(null_bytes_to_write, dtype=np.uint8)
        #Scrivi un array di zeri per riempire i primi 4096 bytes\
        outfile.write(bytearray(array_to_write))

        #The first 4096 bytes containing the are written
        #Now, let's start writeing the real data

        block_size = 1024 # dim of each FFT
        saving_steps = int(data_length / block_size)

        print("Steps: " + str(saving_steps))

        outfile = open(outfilename, 'ab')
        for i in range(saving_steps):
        #for i in range(N_blocks_to_read):

            # print("Saving Step: " + str(i))
            array_buffer = data[i*block_size:(i+1)*block_size]
            #array_buffer = data[i]
            outfile.write(bytearray(array_buffer))

        outfile.close()

    #return the dictionary
    @property
    def get_dict(self):
        return self.dict_properties

    #return the central frequency value
    @property
    def get_cf(self):
        return self.freq_MHz

    #return the bandwidth value
    @property
    def get_bw(self):
        return self.bw_MHz

    #return the sampling time value
    @property
    def get_st(self):
        return self.t_samp_us

    #return the file abspath
    @property
    def get_fabspath(self):
        return self.file_abspath

    #return the file nameonly
    @property
    def get_fnameonly(self):
        return self.file_nameonly

    #return the file basename
    @property
    def get_fbasename(self):
        return self.file_basename,

    #return the file extention
    @property
    def get_fextention(self):
        return self.file_extension

    '''
    #return a generic value
    @property
    def value(self):
        return self.value

    #set a generic value
    @value.setter
    def value(self, val):
        self.value = val'''
