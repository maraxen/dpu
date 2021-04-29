#!/usr/bin/env python3

# IMPORTANT #
# Read the README.md file before touching this file.

import os
import io
import sys
import time
import pickle
import shutil
import logging
import numpy as np
import pandas as pd
import tailer as tl
import json
import traceback
from scipy import stats
import socketIO_client

import custom_script
from custom_script import EVOLVER_IP, EVOLVER_PORT, Options

experiment_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), Options.exp_name)
print(experiment_directory)
SIGMOID, LINEAR, THREE_DIMENSION = 'sigmoid', 'linear', '3d'
logger = logging.getLogger('eVOLVER')
eVOLVER_NS = None


def submit_command(data):
    """submit a command to the eVOLVER
    data: a dictionary with param, value, recurring, immediate as keys"""
    return eVOLVER_NS.emit('command', data, namespace='/dpu-evolver')


def last_lines(fpath, length):
    """efficiently places the end of a file into a dataframe
    fPath: path to file
    length: number of lines to try and recall"""
    file = open(fpath)
    last_lines_ = tl.tail(file, length)  # reads last x lines of file
    file.close()
    data = pd.read_csv(io.StringIO('\n'.join(last_lines_)), header=None)
    return data


def first_lines(fpath, length):
    """efficiently places the beginning of a file into a data frame
    fPath path to file
    length: number of lines to try and recall"""
    file = open(fpath)
    firstlines = tl.head(file, length)  # reads first x lines of file
    file.close()
    data = pd.read_csv(io.StringIO('\n'.join(firstlines)), header=None)
    return data


class Vial(object):
    """
    This should be carefully implemented, but this allows variables to be stored and read directly from vials themselves
    which can greatly simplify processing multiplex vial management.
    The scripting has been set-up to take these objects as arguments (rather than a vial number directly)
    """
    # Turbidostat variables
    initial_dilution_time = 0
    last_pump = 0
    od_set = Options.lower_threshold
    average_OD = 0

    def __init__(self, position, vial_volume, **kwargs):
        self.position = position  # position in the array, int 0-15
        self.vial_volume = vial_volume  # volume of media, mL
        self.__dict__.update(kwargs)  # have not tested this implementation in experiment but debugging shows it works

    def restart(self):
        self.initial_dilution_time = Pumps.initial_values(VIALS, fetch_time=True)[self.position]
        self.last_pump = Pumps.last_set_values(VIALS, fetch_time=True)[self.position]
        self.od_set = OD.initial_values(VIALS, fetch_time=False)[self.position]
        self.average_OD = None


# Define the parameters for the vials
VIALS = [
    Vial(x,
         Options.vial_volume,
         lower_threshold=Options.lower_threshold,
         upper_threshold=Options.upper_threshold)
    for x in range(16)
]
# if options.lower_threshold is an array of different values: lower_threshold = options.lower_threshold[x]


class Calibration(object):
    """Object to handle calibration processing"""

    def __init__(self, parameter, cal, **kwargs):
        self.parameter = parameter  # similar to Component.name, acts as variable for saving and processing
        self.cal = cal  # the opened calibration.json
        self.__dict__.update(kwargs)

    # returns the type of calibration relevant to transforming the data
    def fit(self):
        return self.cal['type']

    # returns the calibration coefficients for a particular vial
    # vial: Vial object
    def coeffs(self, vial):
        return self.cal['coefficients'][vial.position]

    # returns the path to the calibration file
    def cal_path(self):
        save_path = os.path.dirname(os.path.realpath(__file__))
        cal_path = os.path.join(save_path, f'{self.parameter}_cal.json')
        return cal_path

    # returns
    def params(self):
        return self.cal['params']


class Component(object):
    data = None
    is_set = None
    is_read = None
    is_held = None
    standard = None
    """The Main refactored feature. Components facilitate interacting with elements on the eVOLVER.
    They allow for standard and can be rapidly deployed for the inclusion and management of new components.
    
    Whether a method requires a set of vials to be passed or a single is noted. Inputs termed
    vials are Vial objects unless otherwise noted.
    
    if you still need to view results, you can key into the vial
    
    example code:
    od_log = OD.pull_log(vials)
    for vial in vials:
        vial_od_readings = od_log[vial.position]
    """

    def __init__(self, name, **kwargs):
        self.name = name  # parameter name for data handling, string, but processing is done via f-strings anyway
        self.__dict__.update(kwargs)

    # returns path to a vial config, where input events are recorded
    # vial: single Vial object
    def vial_config(self, vial):
        return os.path.join(experiment_directory, f'{self.name}_config', f'vial{vial.position}_{self.name}_config.txt')

    # returns path to a vial log, where readings are recorded
    # vial: single Vial object
    def vial_log(self, vial):
        return os.path.join(experiment_directory, f'{self.name}', f'vial{vial.position}_{self.name}.txt')

    # returns a dataframe with time as the index and the vials as columns
    # vials: set of Vial objects
    def pull_log(self, vials):
        idx = pd.read_csv(os.path.join(experiment_directory, f'{self.name}', f'vial0_{self.name}.txt'), header=0)
        log = pd.DataFrame(index=idx[idx.columns[0]], columns=[x for x in range(len(vials))])
        for vial in vials:
            readings = pd.read_csv(self.vial_log(vial=vial), header=0)
            log[vial.position] = readings[readings.columns[1]]
        return log

    # fetch the most recently recorded values for a set of vials
    # vials: set of Vial objects
    # fetch_time: bool relating to whether to return elapsed times
    # lines: optional, number of lines to attempt to recall
    def last_recorded_values(self, vials, fetch_time=False, *lines):
        last_values = []
        column = 1
        val_num = 1
        if lines is not None:
            for line in lines:
                val_num = line
        if fetch_time:
            column = 0
        for vial in vials:
            data = last_lines(self.vial_log(vial), val_num + 4)
            last_values.append(data.tail(val_num)[column])
        return last_values

    # fetch the most recently input values for a set of vials
    # can also fetch the time if wanted
    # vials: set of Vial objects
    # fetch_time: bool relating to whether to return elapsed times
    # lines: optional, number of lines to attempt to recall
    def last_set_values(self, vials, fetch_time=False, *lines):
        column = 1
        val_num = 1
        if lines is not None:
            for line in lines:
                val_num = line
        if fetch_time:
            column = 0
        last_values = []
        for vial in vials:
            data = last_lines(self.vial_config(vial), val_num + 2)
            last_values.append(float(data.tail(val_num)[column]))
        return last_values

    # vials: set of Vial objects
    # fetch_time: bool relating to whether to return elapsed times
    def initial_set_values(self, vials, fetch_time=False):
        column = 1
        if fetch_time:
            column = 0
        first_values = []
        for vial in vials:
            data = first_lines(self.vial_config(vial), 2)
            first_values.append(float(data.head(2)[1, column]))
        return first_values

    # returns a list of the initial recorded values for a sleeve component
    # vials: set of Vial objects
    # fetch_time: bool relating to whether to return elapsed times
    def initial_values(self, vials, fetch_time=False):
        column = 1
        if fetch_time:
            column = 0
        first_values = []
        for vial in vials:
            data = first_lines(self.vial_log(vial), 2)
            first_values.append(float(data.head(2)[column][1]))
        return first_values

    # update the configuration, for recording inputs
    # val can include commas, importantly
    # vial: single Vial object
    def update_config(self, vial, val, elapsed_time):
        text_file = open(self.vial_config(vial), 'w')
        text_file.write(f'{elapsed_time},{val}\n')
        text_file.close()
        return

    # for creating files at the start of experiment, currently unimplemented via Components
    # vials: set of Vial objects
    def create_file(self, vials, directory=None, defaults="0,0"):
        if directory is None:
            directory = self.name
        # creates the data directory if it doesn't already exist
        if not os.path.exists(os.path.join(experiment_directory, f'{directory}')):
            os.makedirs(os.path.join(experiment_directory, f'{directory}'))
        for vial in vials:
            f_name = f"vial{vial.position}_{directory}.txt"
            f_path = os.path.join(experiment_directory, f'{directory}', f_name)
            text_file = open(f_path, "w")
            text_file.write(f"Experiment: {Options.exp_name} vial {vial.position}, {time.time()} \n{defaults} \n")
            text_file.close()
        return

    # for submitting a recurring command
    # stir and temperature commands
    # command is set of values (strings or ints), each position in the index references
    # the corresponding sleeve on the evolver
    def recurring_command(self, command):
        data = {'param': f'{self.name}', 'value': command,
                'immediate': False, 'recurring': True}
        logger.debug(f'{self.name} command: {data}')
        return submit_command(data)

    # for submitting an immediate command
    # light and pump commands
    # command is set of values (strings or ints), each position in the index references
    # the corresponding sleeve on the evolver
    def dynamic_command(self, command):
        logger.debug(f'{self.name} command: {command}')
        data = {'param': f'{self.name}', 'value': command,
                'recurring': False, 'immediate': True}
        return submit_command(data)

    # fetch calibrations from the eVOLVER
    @staticmethod
    def request_calibrations():
        logger.debug('requesting active calibrations')
        eVOLVER_NS.emit('getactivecal',
                        {}, namespace='/dpu-evolver')

    # returns Calibration for the associated Component or requests calibrations from the eVOLVER
    # this can be interacted with using Calibration object methods
    def calibration(self) -> Calibration:
        save_path = os.path.dirname(os.path.realpath(__file__))
        if self.name == 'pumps':
            cal_path = os.path.join(save_path, f'{self.name}.txt')
            flow_calibration = first_lines(cal_path, 2)
            if len(flow_calibration.iloc[0]) == 16:
                flow_rate = flow_calibration
            else:
                # Currently just implementing influx flow rate
                flow_rate = flow_calibration[0, :]
            return flow_rate
        cal_path = os.path.join(save_path, f'{self.name}_cal.json')
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                cal = json.load(f)
                return Calibration(self.name, cal)
        if not os.path.exists(cal_path):
            logger.warning('Calibrations not received yet, requesting again')
            self.request_calibrations()
            time.sleep(5)
            cal_path = os.path.join(save_path, f'{self.name}_cal.json')
            if os.path.exists(cal_path):
                with open(cal_path) as f:
                    cal = json.load(f)
                    return Calibration(self.name, cal)
            else:
                pass

    # checks if calibrations for a component exist, if not, requests them from the eVOLVER
    # currently unimplemented but could envision usefulness in additional applications
    def check_for_calibrations(self):
        result = True
        if not os.path.exists(self.calibration().cal_path()):
            # log and request again
            logger.warning('Calibrations not received yet, requesting again')
            self.request_calibrations()
            result = False
        return result

    # upon receiving calibrations, creates raw data directories
    def on_activecalibrations(self, data):
        print('Calibrations received')
        f_path = f'{self.name}_cal.json'
        for calibration in data:
            for fit in calibration['fits']:
                if fit['active']:
                    with open(f_path, 'w') as f:
                        json.dump(fit, f)
                    # Create raw data directories and files for params needed
                    for param in fit['params']:
                        self.create_file(VIALS, directory=param + '_raw')
                    break

    # save readings from the eVOLVER
    # vials: set of Vial objects
    def save_data(self, data, elapsed_time, vials):
        if len(data) == 0:
            return
        for vial in vials:
            file_name = f"vial{vial.position}_{self.name}.txt"
            file_path = os.path.join(experiment_directory, f'{self.name}', file_name)
            text_file = open(file_path, "a+")
            text_file.write(f"{elapsed_time},{data[vial.position]}\n")
            text_file.close()
        return

    # save raw readings
    # vials: set of Vial objects
    def save_raw_data(self, data, elapsed_time, vials):
        if len(data) == 0:
            return
        if not os.path.isdir(os.path.join(experiment_directory, f'{self.name}_raw')):
            os.makedirs(os.path.join(experiment_directory, f'{self.name}_raw'))
        for vial in vials:
            file_name = f"vial{vial.position}_{self.name}_raw.txt"
            file_path = os.path.join(experiment_directory, f'{self.name}_raw', file_name)
            text_file = open(file_path, "a+")
            text_file.write(f"{elapsed_time},{data[vial.position]}\n")
            text_file.close()
        return

    # transform raw readings to calibration, returns the transformed data for a set of vials
    # vials: set of Vial objects
    def transform_data(self, data, vials):
        calibrate = self.calibration()
        data_2 = None
        set_data = None
        if self.is_set:
            set_data = data['config'].get(self.name, None).get('value', None)
        if calibrate.fit == '3d':
            data_2 = data['data'].get(calibrate.params()[1], None)
        data = data['data'].get(calibrate.params()[0], None)

        if data is None or (self.is_set and set_data is None):
            print('Incomplete data received, Error with measurement')
            logger.error('Incomplete data received, error with measurements')
            return None
        if 'NaN' in data or (self.is_set and 'NaN' in set_data):
            print('NaN received, Error with measurement')
            logger.error('NaN received, error with measurements')
            return None

        data = np.array([float(x) for x in data])
        data_df = pd.DataFrame(columns=['data', 'transformed', 'data2'])
        data_df['data'] = data

        if calibrate.fit == THREE_DIMENSION:
            data_2 = np.array([float(x) for x in data_2])
            data_df['data2'] = data_2

        last_set = []
        if self.is_set:
            last_set = Options.temp_initial
            # right now the only set and held one is temperature, so its only relevant for that one
            last_set = np.array(last_set)
        for vial in vials:
            coefficients = calibrate.coeffs(vial)
            try:
                if calibrate.fit() == SIGMOID:
                    # convert raw photodiode data into OD data using calibration curve
                    data[vial.position] = np.real(coefficients[2] -
                                                  ((np.log10((coefficients[1] -
                                                              coefficients[0]) /
                                                             (float(data[vial.position]) -
                                                              coefficients[0]) - 1)) /
                                                   coefficients[3]))
                    if not np.isfinite(data[vial.position]):
                        data[vial.position] = 'NaN'
                        logger.debug(f'{self.name} from vial {vial.position}: {data[vial.position]}')
                    else:
                        logger.debug(f'{self.name} from vial {vial.position}: {data[vial.position]}')
                elif calibrate.fit() == THREE_DIMENSION:
                    data[vial.position] = np.real(coefficients[0] +
                                                  (coefficients[1] * data[vial.position]) +
                                                  (coefficients[2] * data_2[vial.position]) +
                                                  (coefficients[3] * (data[vial.position] ** 2)) +
                                                  (coefficients[4] * data[vial.position] * data_2[vial.position]) +
                                                  (coefficients[5] * (data_2[vial.position] ** 2)))
                elif calibrate.fit() == LINEAR:
                    data[vial.position] = (float(data[vial.position]) *
                                           coefficients[0]) + coefficients[1]
                    logger.debug(f'{self.name} from vial {vial.position}: {data[vial.position]}')
                else:
                    logger.error(f'{self.name} calibration not of supported type!')
                    data[vial.position] = 'NaN'
            except ValueError:
                print(f"{calibrate.parameter} Read Error")
                logger.error(f'{self.name} read error for vial {vial.position}, setting to NaN')
                data[vial.position] = 'NaN'
            if self.is_set and self.is_held:
                try:
                    set_data[vial.position] = (float(set_data[vial.position]) *
                                               coefficients[0]) + coefficients[1]
                    logger.debug(f'set_{self.name} from vial {vial.position}: {set_data[vial.position]}')
                except ValueError:
                    print(f"Set {calibrate.parameter} Read Error")
                    logger.error(f'set {self.name} read error for vial {vial.position}, setting to NaN')
                    set_data[vial.position] = 'NaN'
            # update only if difference with expected
            # value is above 0.2 degrees celsius
        if self.is_held:
            delta_t = np.abs(last_set - data).max()
            if delta_t > 0.2:
                logger.info(f'updating {calibrate.parameter} (max. deltaT is {delta_t})')
                raw_values = [str(int((last_set[vial.position] - calibrate.coeffs(vial)[1])
                                      / calibrate.coeffs(vial)[0])) for vial in vials]
                self.recurring_command(raw_values)

            else:
                logger.debug(f'{self.name} config: {last_set}')
                logger.debug(f'actual {self.name}: {data}')
        data_df['transformed'] = data
        return data


OD = Component('OD', is_read=True, standard=Options.initial_OD)
Temperature = Component(name='temp', is_read=True, is_held=True, is_set=True)
Stir = Component('stir', is_set=True)
Light = Component('light', is_set=True)
Pumps = Component('pump', is_set=True)


class EvolverNamespace(socketIO_client.BaseNamespace):
    start_time = None
    use_standard = False
    OD_initial = None

    def on_connect(self, *args):
        print("Connected to eVOLVER as client")
        logger.info('connected to eVOLVER as client')

    def on_disconnect(self, *args):
        print("Disconnected from eVOLVER as client")
        logger.info('disconnected to eVOLVER as client')

    def on_reconnect(self, *args):
        print("Reconnected to eVOLVER as client")
        logger.info("reconnected to eVOLVER as client")

    def on_broadcast(self, data):
        logger.debug('broadcast received')
        elapsed_time = round((time.time() - self.start_time) / 3600, 4)
        logger.debug(f'elapsed time: {elapsed_time} hours')
        print(f"{Options.exp_name}: {elapsed_time} Hours")

        # are the calibrations in yet?
        if not OD.check_for_calibrations() or not Temperature.check_for_calibrations():
            logger.warning('calibration files still missing, skipping custom '
                           'functions')
            return

        # apply calibrations and update temperatures if needed
        OD.data = OD.transform_data(data, VIALS)
        Temperature.data = Temperature.transform_data(data, VIALS)

        if OD.data is None or Temperature.data is None:
            logger.error('could not transform raw data, skipping user-'
                         'defined functions')
            return

        # should we "blank" the OD?
        if self.use_standard and self.OD_initial is None:
            logger.info('setting initial OD reading')
            self.OD_initial = OD.data
        elif self.OD_initial is None:
            self.OD_initial = np.zeros(len(VIALS))
        if self.use_standard:
            OD.data = (OD.data - self.OD_initial) + OD.standard

        # save data
        OD.save_data(OD.data, elapsed_time, VIALS)
        Temperature.save_data(Temperature.data, elapsed_time, VIALS)

        for param in OD.calibration().params():
            OD.save_raw_data(data['data'].get(param, []), elapsed_time, VIALS)

        for param in Temperature.calibration().params():
            Temperature.save_raw_data(data['data'].get(param, []), elapsed_time, VIALS)

        # run custom functions
        self.custom_functions(VIALS, elapsed_time)
        # save variables
        self.save_variables(self.start_time, self.OD_initial)

    def stop_all_pumps(self, ):
        data = {'param': 'pump',
                'value': ['0'] * 48,
                'recurring': False,
                'immediate': True}
        logger.info('stopping all pumps')
        self.emit('command', data, namespace='/dpu-evolver')

    def request_calibrations(self):
        logger.debug('requesting active calibrations')
        self.emit('getactivecal',
                  {}, namespace='/dpu-evolver')

    def initialize_exp(self, vials):
        logger.debug('initializing experiment')

        if os.path.exists(experiment_directory):
            logger.info('found an existing experiment')
            exp_continue = None
            while exp_continue not in ['y', 'n']:
                exp_continue = input(
                    'Continue from existing experiment? (y/n): ')
        else:
            exp_continue = 'n'

        if exp_continue == 'n':
            if os.path.exists(experiment_directory):
                exp_overwrite = None
                logger.info('data directory already exists')
                while exp_overwrite not in ['y', 'n']:
                    exp_overwrite = input('Directory already exists. '
                                          'Overwrite with new experiment? (y/n): ')
                if exp_overwrite == 'y':
                    logger.info('deleting existing data directory')
                    shutil.rmtree(experiment_directory)
                else:
                    print('Change experiment name in custom_script.py '
                          'and then restart...')
                    logger.warning(
                        'not deleting existing data directory, exiting')
                    sys.exit(1)

            start_time = time.time()

            OD.request_calibrations()

            logger.debug('creating data directories')

            # make OD file
            OD.create_file(vials)
            # make temperature data file
            Temperature.create_file(vials)
            # make temperature configuration file
            Temperature.create_file(vials, directory='temp_config', defaults=f"0,{Options.temp_initial[0]}")
            # make pump log file [time, pump_duration, average (smoothed) OD].
            # Timed morbidostat uses the third column to track states instead of smoothed OD.
            Pumps.create_file(vials, directory='pump_config', defaults="0,0,0")
            # make OD set file
            OD.create_file(vials, directory='OD_config', defaults="0,Options.upper_threshold")
            OD.create_file(vials, directory='gr', defaults="0,0")

            Stir.recurring_command(Options.stir_initial)

            exp_blank = input('Calibrate vials to standard? (y/n): ')
            if exp_blank == 'y':
                # will do it with first broadcast
                self.use_standard = True
                logger.info('will use initial OD measurement as blank')
            else:
                self.use_standard = False
                self.OD_initial = None  # we can make this what we want
        else:
            # load existing experiment
            pickle_name = f"{Options.exp_name}.pickle"
            pickle_path = os.path.join(experiment_directory, pickle_name)
            logger.info('loading previous experiment data: %s' % pickle_path)
            with open(pickle_path, 'rb') as f:
                loaded_var = pickle.load(f)
            x = loaded_var
            start_time = x[0]
            self.OD_initial = x[1]
            for vial in VIALS:
                vial.restart()

        # copy current custom script to txt file
        backup_filename = f"{Options.exp_name}_{time.time_ns()}.txt"
        shutil.copy('custom_script.py', os.path.join(experiment_directory,
                                                     backup_filename))
        logger.info('saved a copy of current custom_script.py as %s' %
                    backup_filename)

        return start_time

    @staticmethod
    def save_variables(start_time, od_initial):
        # save variables needed for restarting experiment later
        pickle_name = f"{Options.exp_name}.pickle"
        pickle_path = os.path.join(experiment_directory, pickle_name)
        logger.debug(f'saving all variables: {pickle_path}')
        with open(pickle_path, 'wb') as f:
            pickle.dump([start_time, od_initial], f)

    @staticmethod
    def calc_growth_rate(vial_position, gr_start, elapsed_time):
        data = OD.pull_log(VIALS)
        raw_time = data.index
        raw_od = data[vial_position]
        raw_time = raw_time[np.isfinite(raw_od)]
        raw_od = raw_od[np.isfinite(raw_od)]

        # Trim points prior to gr_start
        trim_time = raw_time[np.nonzero(np.where(raw_time > gr_start, 1, 0))]
        trim_od = raw_od[np.nonzero(np.where(raw_time > gr_start, 1, 0))]

        # Take natural log, calculate slope
        log_od = np.log(trim_od)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            trim_time[np.isfinite(log_od)],
            log_od[np.isfinite(log_od)])
        logger.debug(f'growth rate for vial {vial_position}: {slope}')

        # Save slope to file
        file_name = f"vial{vial_position}_gr.txt"
        gr_path = os.path.join(experiment_directory, 'gr', file_name)
        text_file = open(gr_path, "a+")
        text_file.write(f"{elapsed_time},{slope}\n")
        text_file.close()

    def custom_functions(self, vials, elapsed_time):
        # load user script from custom_script.py
        if Options.algo == 'turbidostat':
            custom_script.turbidostat(self, vials, elapsed_time, Options, OD, Pumps, Light)
            """elif Options.algo == 'chemostat':
                custom_script.chemostat(self, data, vials, elapsed_time, options)
            elif Options.algo == 'morbidostat':
                custom_script.morbidostat(self, data, vials, elapsed_time, options)
            elif Options.algo == 'timed_morbidostat':
                custom_script.timed_morbidostat(
                    self, data, vials, elapsed_time, options)
            elif Options.algo == 'old_morbidostat':
                custom_script.old_morbidostat(
                    self, data, vials, elapsed_time, options)"""
        else:
            # try to load the user function
            # if failing report to user
            logger.info(f'user-defined operation mode {Options.algo}')
            try:
                func = getattr(custom_script, Options.algo)
                func(self, vials, elapsed_time)
            except AttributeError:
                logger.error('could not find function %s in custom_script.py' %
                             Options.algo)
                print('Could not find function %s in custom_script.py '
                      '- Skipping user defined functions' %
                      Options.algo)

    def stop_exp(self):
        self.stop_all_pumps()


if __name__ == '__main__':
    print(Options.exp_name)
    print(Options.algo)
    # changes terminal tab title in OSX
    print('\x1B]0;eVOLVER EXPERIMENT: PRESS Ctrl-C TO PAUSE\x07')

    # silence logging until experiment is initialized
    logging.level = logging.CRITICAL + 10

    socketIO = socketIO_client.SocketIO(EVOLVER_IP, EVOLVER_PORT)
    eVOLVER_NS = socketIO.define(EvolverNamespace, '/dpu-evolver')

    # start by stopping any existing chemostat
    eVOLVER_NS.stop_all_pumps()
    #
    eVOLVER_NS.start_time = eVOLVER_NS.initialize_exp(VIALS)

    # logging setup
    if Options.quiet:
        logging.basicConfig(level=logging.CRITICAL + 10)
    else:
        if Options.verbose == 0:
            level = logging.INFO
            logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s] '
                                       '- %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                filename=os.path.join(experiment_directory, 'eVOLVER.log'),
                                level=level)
        elif Options.verbose >= 1:
            level = logging.DEBUG
            logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s] '
                                       '- %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                filename=os.path.join(experiment_directory, 'eVOLVER.log'),
                                level=level)

    reset_connection_timer = time.time()
    while True:
        try:
            # infinite loop
            socketIO.wait(seconds=0.1)
            if time.time() - reset_connection_timer > 3600:
                # reset connection to avoid buildup of broadcast
                # messages (unlikely but could happen for very long
                # experiments with slow dpu code/computer)
                logger.info('resetting connection to eVOLVER to avoid '
                            'potential buildup of broadcast messages')
                socketIO.disconnect()
                socketIO.connect()
                reset_connection_timer = time.time()
        except KeyboardInterrupt:
            try:
                print('Ctrl-C detected, pausing experiment')
                logger.warning('interrupt received, pausing experiment')
                eVOLVER_NS.stop_exp()
                # stop receiving broadcasts
                socketIO.disconnect()
                while True:
                    key = input('Experiment paused. Press enter key to restart '
                                ' or hit Ctrl-C again to terminate experiment')
                    logger.warning('resuming experiment')
                    # no need to have something like "restart_chemo" here
                    # with the new server logic
                    socketIO.connect()
                    break
            except KeyboardInterrupt:
                print('Second Ctrl-C detected, shutting down')
                logger.warning('second interrupt received, terminating '
                               'experiment')
                eVOLVER_NS.stop_exp()
                print('Experiment stopped, goodbye!')
                logger.warning('experiment stopped, goodbye!')
                break
        except Exception as e:
            logger.critical('exception %s stopped the experiment' % str(e))
            print('error "%s" stopped the experiment' % str(e))
            traceback.print_exc(file=sys.stdout)
            eVOLVER_NS.stop_exp()
            print('Experiment stopped, goodbye!')
            logger.warning('experiment stopped, goodbye!')
            break
    socketIO.connect()
    eVOLVER_NS.stop_exp()
