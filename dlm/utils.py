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
	OFF = BOLD + '\033[0,0m'

def pprint(message, print_target="stdout"):
	assert_value(print_target, ["stdout", "stderr"])
	if print_target == "stderr":
		sys.stderr.write(message)
	else:
		sys.stdout.write(message)
	
def pprintln(message, print_target):
	pprint(message + "\n", print_target)

def error(message):
	sys.stderr.write(BColors.BFAIL + "[ERROR] " + BColors.ENDC + message + "\n")
	sys.exit()

def warning(message):
	sys.stderr.write(BColors.BWARNING + "[WARNING] " + BColors.ENDC + message + "\n")
	
def info(message):
	sys.stderr.write(BColors.BOKBLUE + "[INFO] " + BColors.ENDC + message + "\n")
	
def usage(message):
	sys.stderr.write(BColors.BOLD + "[USAGE] " + BColors.ENDC + message + "\n")
	sys.exit()

def exception():
	sys.stderr.write(BColors.BFAIL + "[ERROR] " + BColors.ENDC + str(sys.exc_info()[0].mro()[0].__name__) + ": " + sys.exc_info()[1].message + "\n")
	sys.exit()
	
#-----------------------------------------------------------------------------------------------------------#
	
def xassert(condition, message):
	if not condition:
		error(message)

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

def set_theano_device(device):
	import sys
	xassert(device == "cpu" or device == "gpu", "The device can only be 'cpu' or 'gpu'")
	xassert(sys.modules.has_key('theano') == False, "dlm.utils.set_theano_device() function cannot be called after importing theano")
	os.environ['THEANO_FLAGS'] = 'device=' + device
	os.environ['THEANO_FLAGS'] += ',force_device=True'
	os.environ['THEANO_FLAGS'] += ',floatX=float32'
	os.environ['THEANO_FLAGS'] += ',print_active_device=False'
	os.environ['THEANO_FLAGS'] += ',mode=FAST_RUN'
	os.environ['THEANO_FLAGS'] += ',nvcc.fastmath=True' # makes div and sqrt faster at the cost of precision
	try:
		import theano
	except EnvironmentError:
		exception()
	if theano.config.device == "gpu":
		info(
			"Device: " + theano.config.device.upper() + " "
			+ str(theano.sandbox.cuda.active_device_number())
			+ " (" + str(theano.sandbox.cuda.active_device_name()) + ")"
		)
	else:
		info("Device: " + theano.config.device.upper())




