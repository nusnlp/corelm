import subprocess as sub
import sys
import os, errno
	
#-----------------------------------------------------------------------------------------------------------#

def __shell(command):
	return sub.Popen(command, shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
	
# Currently the best
def capture(command):
	out, err, code = capture_all(command)
	assert (code == 0), "Failed to run the command: " + command
	return out

# Good, if more info is needed
def capture_all(command):
	p = __shell(command)
	output, err = p.communicate()
	return output, err, p.returncode
	
# Better to avoid
def capture_no_assert(command):
	p = __shell(command)
	return p.stdout.read()
	
# Not well-tested, but should be good
def capture_output(command):
	try:
		eval("sub.check_output")
	except:
		error("subprocess check_output function is not supported in this python version:" + version())
	output = sub.check_output(command, shell=True)
	return output

#-----------------------------------------------------------------------------------------------------------#

# Dummy object for holding other objects
class Object(object):
    pass
	
#-----------------------------------------------------------------------------------------------------------#
	
import re

class BColors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	WHITE = '\033[37m'
	YELLOW = '\033[33m'
	GREEN = '\033[32m'
	BLUE = '\033[34m'
	CYAN = '\033[36m'
	RED = '\033[31m'
	MAGENTA = '\033[35m'
	BLACK = '\033[30m'
	BHEADER = BOLD + '\033[95m'
	BOKBLUE = BOLD + '\033[94m'
	BOKGREEN = BOLD + '\033[92m'
	BWARNING = BOLD + '\033[93m'
	BFAIL = BOLD + '\033[91m'
	BUNDERLINE = BOLD + '\033[4m'
	BWHITE = BOLD + '\033[37m'
	BYELLOW = BOLD + '\033[33m'
	BGREEN = BOLD + '\033[32m'
	BBLUE = BOLD + '\033[34m'
	BCYAN = BOLD + '\033[36m'
	BRED = BOLD + '\033[31m'
	BMAGENTA = BOLD + '\033[35m'
	BBLACK = BOLD + '\033[30m'
	
	@staticmethod
	def cleared(s):
		return re.sub("\033\[[0-9][0-9]?m", "", s)

def red(message):
	return BColors.RED + str(message) + BColors.ENDC

def b_red(message):
	return BColors.BRED + str(message) + BColors.ENDC

def blue(message):
	return BColors.BLUE + str(message) + BColors.ENDC

def b_yellow(message):
	return BColors.BYELLOW + str(message) + BColors.ENDC

def green(message):
	return BColors.GREEN + str(message) + BColors.ENDC

def b_green(message):
	return BColors.BGREEN + str(message) + BColors.ENDC
	
#-----------------------------------------------------------------------------------------------------------#
	
def xassert(condition, message):
	if not condition:
		import dlm.io.logging as L
		L.error(message)

def assert_value(value, valid_values):
	assert type(valid_values) == list, "valid_values must be a list, given: " + str(type(valid_values))
	assert value in valid_values, "Invalid value: " + str(value) + " is not in " + str(valid_values)
	
def version():
	return '.'.join(map(str, sys.version_info)[0:3])
	
#-----------------------------------------------------------------------------------------------------------#

def prepend_to_file(file_name, text):
	with open(file_name, "r+") as f:
		old = f.read()
		f.seek(0)
		f.write(text + old)
	
def append_to_file(file_name, text):
	with open(file_name, "a") as f:
		f.write(text)

def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise

def num_lines(path):
	return sum(1 for line in open(path))

#-----------------------------------------------------------------------------------------------------------#

def get_all_windows(input_list, window_size):
	if window_size <= 1:
		return input_list
	output = []
	for i in range(len(input_list) - window_size + 1):
		output.append(input_list[i:i+window_size])
	return output

#-----------------------------------------------------------------------------------------------------------#

def is_gpu_free(gpu_id):
	out = capture('nvidia-smi -i ' + str(gpu_id)).strip()
	tokens = out.split('\n')[-2].split()
	return ' '.join(tokens[1:5]) == 'No running processes found'

def set_theano_device(device, threads):
	import sys
	import dlm.io.logging as L
	xassert(device == "cpu" or device.startswith("gpu"), "The device can only be 'cpu', 'gpu' or 'gpu<id>'")
	if device.startswith("gpu") and len(device) > 3:
		try:
			gpu_id = int(device[3:])
			if not is_gpu_free(gpu_id):
				L.warning('The selected GPU (GPU' + str(gpu_id) + ') is apparently busy.')
		except ValueError:
			L.error("Unknown GPU device format: " + device)
	if device.startswith("gpu"):
		L.warning('Running on GPU yields non-deterministic results.')
	xassert(sys.modules.has_key('theano') == False, "dlm.utils.set_theano_device() function cannot be called after importing theano")
	os.environ['OMP_NUM_THREADS'] = str(threads)
	os.environ['THEANO_FLAGS'] = 'device=' + device
	os.environ['THEANO_FLAGS'] += ',force_device=True'
	os.environ['THEANO_FLAGS'] += ',floatX=float32'
	os.environ['THEANO_FLAGS'] += ',warn_float64=warn'
	os.environ['THEANO_FLAGS'] += ',cast_policy=numpy+floatX'
	#os.environ['THEANO_FLAGS'] += ',allow_gc=True'
	os.environ['THEANO_FLAGS'] += ',print_active_device=False'
	os.environ['THEANO_FLAGS'] += ',exception_verbosity=high'		# Highly verbose debugging
	os.environ['THEANO_FLAGS'] += ',mode=FAST_RUN'
	os.environ['THEANO_FLAGS'] += ',nvcc.fastmath=False' 			# True: makes div and sqrt faster at the cost of precision, and possible bugs
	#os.environ['THEANO_FLAGS'] += ',optimizer_including=cudnn' 	# Comment out if CUDNN is not available
	try:
		import theano
	except EnvironmentError:
		L.exception()
	global logger
	if theano.config.device == "gpu":
		L.info(
			"Device: " + theano.config.device.upper() + " "
			+ str(theano.sandbox.cuda.active_device_number())
			+ " (" + str(theano.sandbox.cuda.active_device_name()) + ")"
		)
	else:
		L.info("Device: " + theano.config.device.upper())

#-----------------------------------------------------------------------------------------------------------#

def print_args(args):
	import dlm.io.logging as L
	L.info("Arguments:")
	items = vars(args)
	for key in sorted(items.keys(), key=lambda s: s.lower()):
		value = items[key]
		if not value:
			value = "None"
		L.info("  " + key + ": " + BColors.MAGENTA + str(items[key]) + BColors.ENDC)

def curr_time():
	import time
	t = time.localtime()
	return '%i-%i-%i-%ih-%im-%is' % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

def curr_version():
	import dlm.io.logging as L
	info_path = os.path.dirname(sys.argv[0]) + '/.git/refs/heads/master'
	if os.path.exists(info_path):
		with open(info_path, 'r') as info_file:
			return info_file.next().strip()
	L.warning('Unable to read current version.')
	return None
