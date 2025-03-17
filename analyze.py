import argparse
import csv
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import sys
import warnings

import imageops
import keyence
import util

# for windows consoles (e.g. git bash) to work properly
try:
	sys.stdin.reconfigure(encoding='utf-8')
	sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
	pass # sys.stdout has been wrapped, but should already support utf-8

channel_main_ototox = int(util.get_config('channel_main_ototox'))
channel_subtr_ototox = int(util.get_config('channel_subtr_ototox'))
replacement_delim = util.get_config('filename_replacement_delimiter')
replacement_brfld = util.get_config('filename_replacement_brightfield_ototox').split(replacement_delim)
replacement_mask = util.get_config('filename_replacement_mask_ototox').split(replacement_delim)
replacement_subtr = util.get_config('filename_replacement_subtr_ototox').split(replacement_delim)

class Image:
	channel = channel_main_ototox
	channel_subtr = channel_subtr_ototox
	particles = True
	replacement_brfld = replacement_brfld
	replacement_mask = replacement_mask
	replacement_subtr = replacement_subtr

	def __init__(self, filename, group, debug=0):
		self.fl_filename = filename
		self.bf_filename = filename.replace(self.replacement_brfld[0], self.replacement_brfld[1])
		self.subtr_filename = filename.replace(self.replacement_subtr[0], self.replacement_subtr[1])

		match = re.search(r'([a-zA-Z0-9]+)_XY([0-9][0-9])_', filename)
		if not match:
			raise UserError('Filename %s missing needed xy information' % filename)

		self.plate = match.group(1)
		self.xy = int(match.group(2))

		self.group = group
		self.debug = debug

		self.bf_img = None
		self.bf_metadata = None
		self.fl_img = None
		self.fl_metadata = None
		self.subtr_img = None
		self.mask = None
		self.normalized_value = None
		self.value = None

	def get_bf_img(self):
		if self.bf_img is None:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", UserWarning)
				self.bf_img = imageops.read(self.bf_filename, np.uint16)
		return self.bf_img

	def get_bf_metadata(self):
		if self.bf_metadata is None:
			self.bf_metadata = keyence.extract_metadata(self.bf_filename)
		return self.bf_metadata

	def get_fl_img(self):
		if self.fl_img is None:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", UserWarning)
				self.fl_img = imageops.read(self.fl_filename, np.uint16, self.channel)
		return self.fl_img

	def get_fl_metadata(self):
		if self.fl_metadata is None:
			self.fl_metadata = keyence.extract_metadata(self.fl_filename)
		return self.fl_metadata

	def get_mask(self):
		if self.mask is None:
			self.mask = imageops.get_fish_mask(
				self.get_bf_img(), self.get_fl_img(), particles=self.particles,
				silent=self.debug < 1, verbose=self.debug >= 2,
				v_file_prefix='{}_XY{:02d}'.format(self.plate, self.xy),
				mask_filename=self.fl_filename.replace(
					self.replacement_mask[0], self.replacement_mask[1]),
				subtr_img=self.get_subtr_img()
			)
		return self.mask

	def get_raw_value(self):
		if self.value is None:
			fl_img_masked = imageops.apply_mask(self.get_fl_img(), self.get_mask())
			score = imageops.score(fl_img_masked)
			self.value = score if score > 0 else np.nan
		return self.value

	def get_subtr_img(self):
		if self.subtr_img is None:
			if os.path.isfile(self.subtr_filename):
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", UserWarning)
					self.subtr_img = imageops.read(
						self.subtr_filename, np.uint16, self.channel_subtr)
			else:
				self.subtr_img = []
		return self.subtr_img

	def normalize(self, control_values, cap):
		try:
			val = float(self.get_raw_value() * 100 // control_values[self.plate])
			if cap > 0:
				self.normalized_value = val if val < cap else np.nan
			else:
				self.normalized_value = val
		except ZeroDivisionError:
			print('ERROR: Plate', self.plate, 'group', self.group, 'with value',
				self.get_raw_value(), 'has control value', control_values[self.plate])
			self.normalized_value = np.nan
		return self

class UserError(ValueError):
	pass

def chart(results, chartfile, scale='linear'):
	with sns.axes_style(style='whitegrid'):
		data = pd.DataFrame({
			'brightness': [value for values in results.values() for value in values],
			'group': [key for key, values in results.items() for _ in values],
		})

		fig = plt.figure(figsize=(12, 8), dpi=100)
		ax = sns.swarmplot(x='group', y='brightness', data=data)
		ax.set_yscale(scale)
		if scale == 'linear':
			ax.set_ylim(bottom=0)
		sns.boxplot(x='group', y='brightness', data=data, showbox=False, showcaps=False,
			showfliers=False, whiskerprops={'visible': False})
		plt.xticks(rotation=90)
		plt.tight_layout()
		plt.savefig(chartfile)

def get_schematic(platefile, target_count, plate_ignore=[], flat=True):
	if not platefile:
		return keyence.LAYOUT_DEFAULT

	if '' not in plate_ignore:
		plate_ignore.append('')

	with open(platefile, encoding='utf8', newline='') as f:
		schematic = [
			[_clean(well) for well in row if well not in plate_ignore] for row in csv.reader(f)
		]

	count = sum([len(row) for row in schematic])
	if count != target_count:# try removing first row and first column, see if then it matches up
		del schematic[0]
		for row in schematic:
			del row[0]
		count = sum([len(row) for row in schematic])
		if count != target_count:
			raise UserError(
				f'Schematic does not have same number of cells ({count}) as images provided ' +
					f'({target_count})')

	return schematic if not flat else [well for row in schematic for well in row]

def main(imagefiles, cap=-1, chartfile=None, debug=0, group_regex='.*', platefile=None,
		plate_control=['B'], plate_ignore=[], silent=False):
	results = {}

	schematic = get_schematic(platefile, len(imagefiles), plate_ignore)
	groups = list(dict.fromkeys(schematic))# deduplicated copy of `schematic`
	images = quantify(imagefiles, plate_control, cap=cap, debug=debug, group_regex=group_regex,
		schematic=schematic)

	pattern = re.compile(group_regex)
	for group in groups:
		if group in plate_control or pattern.search(group):
			relevant_values = [img.normalized_value for img in images if img.group == group]
			results[group] = relevant_values
			if not silent:
				with warnings.catch_warnings():
					warnings.simplefilter('ignore', RuntimeWarning)
					print(group, np.nanmedian(relevant_values), relevant_values)

	if chartfile:
		chart(results, chartfile)

	return results

def quantify(imagefiles, plate_control=['B'], cap=-1, debug=0, group_regex='.*', schematic=None):
	pattern = re.compile(group_regex)
	images = [Image(filename, group, debug) for filename, group in zip(imagefiles, schematic)
		if group in plate_control or pattern.search(group)]
	control_values = _calculate_control_values(images, plate_control)
	return [image.normalize(control_values, cap) for image in images]

def _calculate_control_values(images, plate_control):
	ctrl_imgs = [img for img in images if img.group in plate_control]
	ctrl_vals = {}

	for plate in np.unique([img.plate for img in ctrl_imgs]):
		ctrl_results = np.array([img.get_raw_value() for img in ctrl_imgs if img.plate == plate])
		ctrl_vals[plate] = float(np.nanmedian(ctrl_results))

	if not ctrl_vals:
		raise UserError(
			'No control wells found. Please supply a --plate-control, or modify the given value.')

	return ctrl_vals

def _clean(s):
	return ''.join(c for c in s if c.isprintable()).strip()

#
# main
#

def set_arguments(parser):
	parser.add_argument('imagefiles',
		nargs='+',
		help='The absolute or relative filenames where the relevant images can be found.')
	parser.add_argument('-ch', '--chartfile',
		help='If supplied, the resulting numbers will be charted at the given filename.')

	parser.add_argument('-p', '--platefile',
		help='CSV file containing a schematic of the plate from which the given images were '
			'taken. Row and column headers are optional. The cell values are essentially just '
			'arbitrary labels: results will be grouped and charted according to the supplied '
			'values.')
	parser.add_argument('-pc', '--plate-control',
		default=['B'],
		nargs='*',
		help=('Labels to treat as the control condition in the plate schematic. These wells are '
			'used to normalize all values in the plate for more interpretable results. Any number '
			'of values may be passed.'))
	parser.add_argument('-pi', '--plate-ignore',
		default=[],
		nargs='*',
		help=('Labels to ignore (treat as null/empty) in the plate schematic. Empty cells will '
			'automatically be ignored, but any other null values (e.g. "[empty]") must be '
			'specified here. Any number of values may be passed.'))

	parser.add_argument('-g', '--group-regex',
		default='.*',
		help=('Pattern to be used to match group names that should be included in the results. '
			'Matched groups will be included, groups that don\'t match will be ignored. Control '
			'wells will always be included regardless of whether they match.'))

	parser.add_argument('-c', '--cap',
		default=-1,
		type=int,
		help=('Exclude well values larger than the given integer, expressed as a percentage of '
			'the median control value.'))

	parser.add_argument('-d', '--debug',
		action='count',
		default=0,
		help=('Indicates intermediate processing images should be output for troubleshooting '
			'purposes. Including this argument once will yield one intermediate image per input '
			'file, twice will yield several intermediate images per input file.'))
	parser.add_argument('-s', '--silent',
		action='store_true',
		help=('If present, printed output will be suppressed. More convenient for programmatic '
			'execution.'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description=('Analyzer for images of whole zebrafish with stained neuromasts, for the '
			'purposes of measuring hair cell damage. Reports values relative to control.'))

	set_arguments(parser)

	args = parser.parse_args(sys.argv[1:])
	args_dict = vars(args)
	try:
		main(**args_dict)
	except UserError as ue:
		print('Error:', ue)
		sys.exit(1)
