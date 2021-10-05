import configparser
import os

_config = None
_section = 'Main'

def get_config(setting, fallback=None):
	global _config
	if _config == None:
		_config = configparser.ConfigParser()
		_config.read(f'{_get_here()}/config.ini')
		_config.read(f'{_get_here()}/config-ext.ini')
	return _config[_section].get(setting, fallback)

def _get_here():
	script = sys.argv[0] if __name__ == '__main__' else __file__
	return os.path.dirname(os.path.realpath(script))