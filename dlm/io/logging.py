import sys
import dlm.utils as U

file_path = None
quiet = False

def set_file_path(path):
	global file_path
	file_path = path
	log_file = open(file_path, 'w') # reset the file
	log_file.close()
	info('Log file: ' + path)

def error(message):
	stderr = U.BColors.BFAIL + "[ERROR] " + U.BColors.ENDC + message + "\n"
	log = "[ERROR] " + U.BColors.cleared(message) + "\n"
	_write(stderr, log)
	sys.exit()

def warning(message):
	stderr = U.BColors.BWARNING + "[WARNING] " + U.BColors.ENDC + message + "\n"
	log = "[WARNING] " + U.BColors.cleared(message) + "\n"
	_write(stderr, log)

def info(message):
	stderr = U.BColors.BOKBLUE + "[INFO] " + U.BColors.ENDC + message + "\n"
	log = "[INFO] " + U.BColors.cleared(message) + "\n"
	_write(stderr, log)

def exception():
	exc = str(sys.exc_info()[0].mro()[0].__name__) + ": " + sys.exc_info()[1].message + "\n"
	stderr = U.BColors.BFAIL + "[ERROR] " + U.BColors.ENDC + exc
	log = "[ERROR] " + exc
	_write(stderr, log)
	sys.exit()

def _write(stderr, log):
	global quiet
	if not quiet:
		sys.stderr.write(stderr)
	global file_path
	if file_path:
		log_file = open(file_path, 'a')
		log_file.write(log)
		log_file.close()
