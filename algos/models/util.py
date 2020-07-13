import pickle as pk
import sys




############################################################
### IO
############################################################
def disp_to_term(msg):
    sys.stdout.write(msg + '\r')
    sys.stdout.flush()

def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print("Pickle file cannot be opened.")
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print('load_pickle failed once, trying again')
        p.close()
        p = open(filename, 'r')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious

def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pk.dump(data_object, pickle_file)
    pickle_file.close()


def conv_output_shape(h_w, kernel_size=3, stride=1, pad=0, dilation=1):
    
    #Utility function for computing output of convolutions
    #takes a tuple of (h,w) and returns a tuple of (h,w)
            
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w
