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
	
def pprint(message, print_target="stdout"):
	assert_value(print_target, ["stdout", "stderr"])
	if print_target == "stderr":
		sys.stderr.write(message)
	else:
		sys.stdout.write(message)
	
def pprintln(message, print_target):
	pprint(message + "\n", print_target)

def error(message):
	sys.stderr.write("[ERROR] " + message + "\n")
	sys.exit()

def warning(message):
	sys.stderr.write("[WARNING] " + message + "\n")
	
def info(message):
	sys.stderr.write("[INFO] " + message + "\n")
	
def usage(message):
	sys.stderr.write("[USAGE] " + message + "\n")
	sys.exit()

def exception():
	sys.stderr.write("[ERROR] " + str(sys.exc_info()[0].mro()[0].__name__) + ": " + sys.exc_info()[1].message + "\n")
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
