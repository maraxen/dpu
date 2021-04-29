#!/usr/bin/env python3
import numpy as np
import pandas as pd
import logging

# logger setup
logger = logging.getLogger(__name__)

EVOLVER_IP = '192.168.1.2'
EVOLVER_PORT = 8081


lit_vials, dark_vials, lit_dark, ascending_vials = [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]
led_values = [0] * 16


class Options(object):
    """This serves for inputs of experiment variables.
    keyword arguments vs assigned variables can universal variables are set every time
    while also facilitating novel contextual applications
    temp_initial, stir_initial, and vial_volume can be single ints for all vials or lists/arrays"""
    def __init__(self, vial_volume, exp_name, pump_wait,
                 time_out, temp_initial, algo, stir_initial, quiet, verbose, **kwargs):
        self.vial_volume = vial_volume  # list/value relating to culture volume for each vial, mL,
        # determined by vial cap straw length
        self.exp_name = exp_name  # name of experiment and what it's associated directory will be titled
        self.time_out = time_out  # (sec) additional amount of time to run efflux pump
        self.pump_wait = pump_wait  # (min) minimum amount of time to wait between pump events
        self.temp_initial = temp_initial  # set of initial temperatures
        self.algo = algo  # which custom function will be performed
        self.stir_initial = stir_initial  # set of initial stir values
        self.quiet = quiet  # logging variable
        self.verbose = verbose  # logging variable
        self.__dict__.update(kwargs)  # have not tested this implementation in experiment but debugging shows it works


Options = Options(vial_volume=25, exp_name='210428_script_test_expt', pump_wait=1, time_out=10,
                  temp_initial=[30] * 16, algo='turbidostat', stir_initial=[8] * 16,
                  quiet=False, verbose=True, stop_after_n_curves=25, to_avg=7,
                  lower_threshold=0.13, upper_threshold=0.16, initial_OD=0.1)
# stop_after_n_curves: set to np.inf to never stop
# or integer value to stop diluting after certain number of growth curves
# to_avg: Number of values to calculate the OD average (for determining action)
# lower/upper_threshold: OD values for the lower and upper bounds of the turbidostat


def turbidostat(evolver, vials, elapsed_time, options, od, pumps, light):
    #  vials fed in can be set to different range (ex. [0,1,2,3]) to only trigger tstat on those vials
    turbidostat_vials = vials
    message = ['--'] * 48     # fluid array message: initialized so that no change is sent

    od_set_times = od.last_set_values(vials, fetch_time=True)  # fetch last dilution time from file
    od_readings = od.last_recorded_values(vials, options.to_avg)
    for vial in turbidostat_vials:  # main loop through each vial
        num_curves = pd.read_csv(od.vial_config(vial)).size / 2  # determine number of dilution cycles

        # pulls the most recent readings given there are enough
        vial.average_OD = float(np.median(np.array(od_readings[vial.position], dtype=float)))

        # Determine whether turbidostat dilutions are needed
        collecting_more_curves = (num_curves <= (options.stop_after_n_curves + 2))

        if od.pull_log(vials)[vial.position].count() != 0:
            # if recently exceeded upper threshold,
            # note end of growth curve in od_set, allow dilutions to occur and growthrate to be measured
            if (vial.average_OD > vial.upper_threshold) and (vial.od_set != vial.lower_threshold):
                od.update_config(vial, vial.lower_threshold, elapsed_time)  # updates the csv
                vial.od_set = vial.lower_threshold
                # calculate growth rate
                evolver.calc_growth_rate(vial.position, od_set_times[vial.position], elapsed_time)

            # if have approx. reached lower threshold, note start of growth curve in od_set
            if (vial.average_OD < (vial.lower_threshold + ((vial.upper_threshold - vial.lower_threshold) / 3))) and (
                    vial.od_set != vial.upper_threshold):
                od.update_config(vial, vial.upper_threshold, elapsed_time)  # updates the csv
                vial.od_set = vial.upper_threshold

            # if need to dilute to lower threshold, then calculate amount of time to pump
            if (vial.average_OD > vial.od_set) and collecting_more_curves:
                time_in = - \
                              (np.log(vial.lower_threshold / vial.average_OD)
                               * vial.vial_volume) / pumps.calibration[vial.position]
                time_in = round(time_in, 2)

                # if sufficient time since last pump, send command to Arduino
                if ((elapsed_time - float(vial.last_pump)) * 60) >= options.pump_wait:
                    logger.info(f'turbidostat dilution for vial {vial.position}')
                    # influx pump
                    message[vial.position] = str(time_in)
                    # efflux pump
                    message[vial.position + 16] = str(time_in + options.time_out)

                    text_file = open(pumps.vial_config(vial.position), "a+")
                    text_file.write(f"{elapsed_time},{time_in},{vial.average_OD}\n")
                    text_file.close()
                    vial.last_pump = elapsed_time
                    if vial.initial_dilution_time == 0:
                        vial.initial_dilution_time = elapsed_time
        else:
            logger.debug(f'not enough OD measurements for vial {vial.position}')
        if vial.initial_dilution_time != 0:
            if vial.position in lit_vials:
                led_values[vial.position] = 4095
                light.dynamic_command(led_values)
            if vial.position in lit_dark:
                if num_curves >= 12:
                    led_values[vial.position] = 4095
                    light.dynamic_command(led_values)
            if vial.position in ascending_vials:
                if (num_curves % 5 == 0) and (num_curves != 0):
                    led_values[vial.position] = 4095 - (4095 / (num_curves/2))
                    light.dynamic_command(led_values)
    # send fluid array command only if we are actually turning on any of the pumps
    if message != ['--'] * 48:
        pumps.dynamic_command(message)


if __name__ == '__main__':
    print('Please run eVOLVER.py instead')
