import nidaqmx
from nidaqmx import constants
import numpy as np
import pandas as pd
from pyTrigger import pyTrigger
import winsound
import matplotlib.pyplot as plt
from ipywidgets import widgets, Output,  Layout, GridspecLayout
from IPython.display import display, clear_output
import json
from tqdm.notebook import tqdm
import time
import math


class VibTesting:
    def __init__(self, task_name, sensor_xlsx, sensor_list, sampling_rate, samps_per_chn, acquisition_time,
                 no_impacts=None,
                 trigger_type='up', trigger_level=10.0, presamples=100, imp_force_lim=0.015, double_imp_force_lim=1,
                 terminal_config=constants.TerminalConfiguration.PSEUDO_DIFF,
                 excitation_source=constants.ExcitationSource.INTERNAL,
                 current_excit_val=0.004, sample_mode=constants.AcquisitionType.CONTINUOUS):
        """
        # TODO: Trenutno samo za IEPE
        # TODO: Check for double impacts -> prilagodi beepe
        # TODO: Force okno
        # TODO: Kontinuirana meritev
        # TODO: Preveri imena naprav v primeru ene same kartice
        nidaqmx constants: https://nidaqmx-python.readthedocs.io/en/latest/constants.html

        # Sensor data parameters
        :param sensor_xlsx: path to the Excel file with sensor data
        :param sensor_list: list of sensors (Serial numbers) or dict of shape {SN: [list of directions (x,y,z)]}

        # General channel configuration parameters
        :param terminal_config: terminal configuration parameter (DEFAULT, DIFF, NRSE,
                                PSEUDO_DIFF, RSE - see nidaqmx constants)
        :param excitation_source: excitation source parameter (EXTERNAL, INTERNAL, NONE - see nidaqmx constants)
        :param current_excit_val: excitation current [A] (float)

        # Sampling parameters
        :param sample_mode: sampling mode (CONTINUOUS, FINITE, HW_TIMED_SINGLE_POINT - see nidaqmx constants)
        :param sampling_rate: sampling rate [Hz] (int)
        :param samps_per_chn: Specifies the number of samples to acquire or generate for each channel in the task if
                              **sample_mode** is **FINITE_SAMPLES**. If **sample_mode** is **CONTINUOUS_SAMPLES**,
                               NI-DAQmx uses this value to determine the buffer size.
        :param acquisition_time: acquisition time [s] (int/float)

        # Trigger configuration parameters
        :param trigger_type: trigger type (up, down or abs - string)
        :param trigger_level: the level to cross, to start trigger (float)
        :param presamples # of presamples

        # Double impact control
        :param imp_force_lim: Limit  value of force derivative for determination of start/end point of the impact
        """
        # Sensor data DataFrame
        self.sensor_df = pd.read_excel(sensor_xlsx)
        self.sensor_list = sensor_list

        # Get all connected NI devices
        system = nidaqmx.system.System.local()
        self.device_list = [_.name for _ in list(system.devices)]

        # Open new task
        try:
            self.task = nidaqmx.task.Task(new_task_name=task_name)
        except nidaqmx.DaqError:
            new_task_name = False
            i = 1
            while not new_task_name:
                try:
                    self.task = nidaqmx.task.Task(new_task_name=task_name + '_{i}')
                    new_task_name = True
                except nidaqmx.DaqError:
                    i += 1
                if i > 5:
                    print('To many tasks generated. Restart kernel to generate new tasks.')
                    break
            print(f"Repeated task name: task name changed to {task_name + '_'}")
        self.excitation_source = excitation_source
        self.current_excit_val = current_excit_val

        # General channel parameters
        self.terminal_config = terminal_config
        # nidaqmx constants - unit conversion
        self.unit_conv = {'mV/g': constants.AccelSensitivityUnits.MILLIVOLTS_PER_G, #'mV/m/s**2': constants.AccelSensitivityUnits.MILLIVOLTS_PER_G, -> mV/m/s**2 ne obstaja -> pretvorba občutljivosti v mV/m/g!
                          'g': constants.AccelUnits.G,
                          'm/s**2': constants.AccelUnits.METERS_PER_SECOND_SQUARED,
                          'mV/N': constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,
                          'N': constants.ForceUnits.NEWTONS}

        # add channels to the task
        self.add_channels()

        # sampling configuration
        self.sampling_rate = sampling_rate
        self.acquisition_time = acquisition_time
        self.task.timing.cfg_samp_clk_timing(rate=self.sampling_rate, sample_mode=sample_mode,
                                             samps_per_chan=samps_per_chn)  # set sampling for the task

        # list all channels
        self.all_channels = [str(_.name) for _ in self.task.ai_channels]
        try:
            self.force_chn_ind = int(np.where(np.array(self.all_channels) == 'force')[0])
        except TypeError:
            print('No force sensor.')
        # trigger configuration
        self.trigger_type = trigger_type
        self.trigger_level = trigger_level
        self.presamples = presamples

        # measurement configuration
        if no_impacts is None:
            no_impacts = 1
        self.no_impacts = no_impacts

        # storing measurements
        self.measurement_array = np.zeros((self.no_impacts,
                                           len(self.all_channels),
                                           int(sampling_rate*acquisition_time)),
                                          dtype=np.float64)  # shape: no_impacts, no_channels, samples

        # Double impact control
        self.imp_force_lim = imp_force_lim
        self.double_imp_force_lim = double_imp_force_lim

        # Measurement info
        try:
            self.meas_info = {'Sampling rate': self.sampling_rate,
                              'Acquisiton time': self.acquisition_time,
                              'Used devices': self.device_list,
                              'Channels': self.all_channels,
                              'Force channel index': self.force_chn_ind}
        except AttributeError:
            self.meas_info = {'Sampling rate': self.sampling_rate,
                              'Acquisiton time': self.acquisition_time,
                              'Used devices': self.device_list,
                              'Channels': self.all_channels}

        # Measurement series variables
        self.meas_file = ''
        self.points_to_measure = []
        self.point_ind = 0
        self.points_measured = []
        self.saved = False

        # Admittance measurement variables
        self.admittance = False
        self.Y = None
        self.dof_data = None
        self.Y_done = None
        self.adm_channels = None
        self.adm_impacts = None
        self.force_chn_ind = None
        self.chn_factors = None
        self.imp_factors = None

    def reset_series_params(self):
        self.point_ind = 0
        self.points_measured = []
        self.saved = False

    # Task generation methods
    def add_channels(self):
        """
        sensors: list of sensors (Serial numbers) or dict of shape {SN: [list of directions ('x','y','z')]}
        task: nidaqmx Task instance
        df: dataframe with sensor data
        """

        device_ind = 1  # pričakovan cDAQ5Mod1
        dev_chn_ind = 0
        sensor_ind = 0

        if type(self.sensor_list) == list:
            for i in self.sensor_list:
                temp_df_ = self.sensor_df[self.sensor_df['SN'].astype(str) == i]
                if temp_df_.empty:
                    raise ValueError(f'Invalid serial number: {i}. Check if the given SN is correct and that it is '
                                     f'included in measurement data file (Merilna oprema.xlsx)')
                for _, chn_ in temp_df_.iterrows():
                    # channel selection
                    try:
                        phys_chn = self.device_list[device_ind] + f'/ai{dev_chn_ind}'
                        chn_name = self.get_chn_name(chn_, sensor_ind)
                        self.new_channel(chn_, physical_channel=phys_chn, name_to_assign_to_channel=chn_name,
                                         min_val=chn_.Min, max_val=chn_.Max,
                                         units=self.unit_conv[chn_['Izhodna enota']],
                                         sensitivity=chn_.Obcutljivost,
                                         sensitivity_units=self.unit_conv[chn_['Enota obcutljivosti']])
                        dev_chn_ind += 1
                    except nidaqmx.DaqError:
                        device_ind += 1
                        dev_chn_ind = 0
                        phys_chn = self.device_list[device_ind] + f'/ai{dev_chn_ind}'
                        chn_name = self.get_chn_name(chn_, sensor_ind)
                        self.new_channel(chn_, physical_channel=phys_chn, name_to_assign_to_channel=chn_name,
                                         min_val=chn_.Min, max_val=chn_.Max,
                                         units=self.unit_conv[chn_['Izhodna enota']],
                                         sensitivity=chn_.Obcutljivost,
                                         sensitivity_units=self.unit_conv[chn_['Enota obcutljivosti']])
                        dev_chn_ind += 1
                sensor_ind += 1
        elif type(self.sensor_list) == dict:
            for sensor_, dir_ in self.sensor_list.items():
                # selecting channels from sensor_df
                temp_df_ = self.sensor_df['SN'].astype(str) == sensor_
                if (temp_df_ == False).all():
                    raise ValueError(f'Invalid serial number: {sensor_}. Check if the given SN is correct and that it '
                                     f'is included in measurement data file (Merilna oprema.xlsx)')
                df_mask = np.zeros_like(temp_df_)
                for i in dir_:
                    df_mask = df_mask | (self.sensor_df['Smer'].astype(str) == i)
                temp_df_ = self.sensor_df[temp_df_ & df_mask]

                for _, chn_ in temp_df_.iterrows():
                    # channel selection
                    try:
                        phys_chn = self.device_list[device_ind] + f'/ai{dev_chn_ind}'
                        chn_name = self.get_chn_name(chn_, sensor_ind)
                        print(chn_name)
                        self.new_channel(chn_, physical_channel=phys_chn, name_to_assign_to_channel=chn_name,
                                         min_val=chn_.Min, max_val=chn_.Max,
                                         units=self.unit_conv[chn_['Izhodna enota']],
                                         sensitivity=chn_.Obcutljivost,
                                         sensitivity_units=self.unit_conv[chn_['Enota obcutljivosti']])
                        # print(sensor, phys_chn, chn_name)
                        dev_chn_ind += 1
                    except nidaqmx.DaqError:
                        device_ind += 1
                        dev_chn_ind = 0
                        phys_chn = self.device_list[device_ind] + f'/ai{dev_chn_ind}'
                        chn_name = self.get_chn_name(chn_, sensor_ind)
                        print(chn_name)
                        self.new_channel(chn_, physical_channel=phys_chn, name_to_assign_to_channel=chn_name,
                                         min_val=chn_.Min, max_val=chn_.Max,
                                         units=self.unit_conv[chn_['Izhodna enota']],
                                         sensitivity=chn_.Obcutljivost,
                                         sensitivity_units=self.unit_conv[chn_['Enota obcutljivosti']])
                        # print( sensor_, phys_chn, chn_name)
                        dev_chn_ind += 1
                    sensor_ind += 1

    def new_channel(self, chn_data, physical_channel, name_to_assign_to_channel, min_val, max_val, units,
                    sensitivity, sensitivity_units):
        # Function adds new channel to the task
        if chn_data['Merjena veličina'] == 'sila':
            self.task.ai_channels.add_ai_force_iepe_chan(physical_channel, name_to_assign_to_channel,
                                                         self.terminal_config,
                                                         min_val, max_val, units, sensitivity, sensitivity_units,
                                                         current_excit_source=self.excitation_source,
                                                         current_excit_val=self.current_excit_val,
                                                         custom_scale_name='')
        else:
            self.task.ai_channels.add_ai_accel_chan(physical_channel, name_to_assign_to_channel, self.terminal_config,
                                                    min_val, max_val, units, sensitivity, sensitivity_units,
                                                    current_excit_source=self.excitation_source,
                                                    current_excit_val=self.current_excit_val,
                                                    custom_scale_name='')

    @staticmethod
    def get_chn_name(chn_, ind_):
        # Function generates channel name string.
        if chn_['Merjena veličina'] == 'sila':
            return 'force'
        else:
            return f'{ind_}{chn_.Smer}'

    # Operational measurement methods
    def start_op_measurement(self, acq_time=None, save_to=None, start_w_button=True):
        # acq_time omejen na celo število
        if acq_time is not None:
            self.acquisition_time = acq_time
        meas_data, pbar = None, None
        if start_w_button:
            start_meas_button = widgets.Button(description='Start measurement')

            display(start_meas_button)

            def start_btn_clicked(B):
                clear_output()
                print('Start btn pressed')
                global pbar
                pbar = self.acquire_op_signal(save_to=save_to)

            start_meas_button.on_click(start_btn_clicked)

        else:
            pbar = self.acquire_op_signal(save_to=save_to)

    def acquire_op_signal(self, save_to):
        self.measurement_array = np.zeros((len(self.all_channels),
                                           int(self.sampling_rate * self.acquisition_time)),
                                          dtype=np.float64)
        try:
            pbar = tqdm(total=self.acquisition_time)
            start_time = time.time()
            i = 1
            self.task.start()
            while True:
                self.measurement_array[:, (i - 1) * self.sampling_rate:i * self.sampling_rate] = np.array(
                    self.task.read(number_of_samples_per_channel=self.sampling_rate, timeout=10.0))
                pbar.update()
                if math.floor(time.time() - start_time) >= self.acquisition_time:
                    pbar.container.children[-2].style.bar_color = 'green'
                    break
                i += 1
            self.task.stop()
            self.plot_op_meas()
            self.save_op_test_results(save_to, pbar)
        except nidaqmx.DaqError:
            clear_output()
            self.task.stop()
            self.acquire_op_signal(save_to=save_to)
        return pbar

    def save_op_test_results(self, save_to, pbar):
        out = Output()
        # Save measurement info
        save_meas_info_choice = widgets.ToggleButtons(
            options=['No', 'Yes'],
            description='Save measurement info: \n',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            style={'description_width': 'initial'}
        )
        # Save measurements
        save_button = widgets.Button(description='Save')
        # Buttons for measurement series

        if save_to is None:
            file_name = widgets.Text(description='Save to: ',
                                     placeholder='Filename',
                                     value='')
            widgets_ = [save_button, save_meas_info_choice, file_name]
        else:
            widgets_ = save_button, save_meas_info_choice
        # widgets layout
        grid = GridspecLayout(2, 2)
        grid[0, 0] = widgets_[0]
        grid[1, :] = widgets_[1]
        if save_to is None:
            grid[0, 1:] = widgets_[2]
        display(grid)

        def save_btn_clicked(B, save_to_=save_to):
            if save_to_ is None:
                save_to_ = str(file_name.value)
                if len(save_to_) == 0:
                    message = 'Enter file name!'
                else:
                    message = f'Measurement saved to \"{save_to_}.npy\"'
                    np.save(f'{save_to_}.npy', self.measurement_array.squeeze())
                    if save_meas_info_choice.value == 'Yes':
                        message += f'\nMeasurement info saved to \"{save_to_}.json\"'
                        json_obj = json.dumps(self.meas_info, indent=4)
                        with open(f'{save_to_}.json', 'w') as meas_data_file:
                            meas_data_file.write(json_obj)
                    pbar.container.children[-2].style.bar_color = 'black'
            else:
                np.save(f'{save_to_}.npy', self.measurement_array.squeeze())
                message = f'Measurements saved to \"{save_to_}.npy\"'
                if save_meas_info_choice.value == 'Yes':
                    message += f'\nMeasurement info saved to \"{save_to_}.json\"'
                    json_obj = json.dumps(self.meas_info, indent=4)
                    with open(f'{save_to_}.json', 'w') as meas_data_file:
                        meas_data_file.write(json_obj)
                pbar.container.children[-2].style.bar_color = 'black'
            with out:
                clear_output()
                print(message)
            display(out)

        save_button.on_click(save_btn_clicked)

    # Impact measurement methods
    def start_impact_test(self, no_impacts=None, save_to=None, series=False):
        """
        :param no_impacts:
        :param save_to: name of the file in which measurements are to be saved.
        :param series: 
        """
        if no_impacts is not None:
            self.no_impacts = no_impacts
            self.measurement_array = np.zeros((self.no_impacts,
                                               len(self.all_channels),
                                               int(self.sampling_rate * self.acquisition_time)),
                                              dtype=np.float64)
        # tqdm
        pbar = tqdm(total=self.no_impacts)

        # prevents error in case of interruption during last measurement
        if not self.task.is_task_done():
            self.task.stop()

        imp = 0
        self.clear_stored_data()
        while imp < self.no_impacts:
            # Beep denoting start of the measurement
            winsound.Beep(410, 180)
            # Signal acquisition
            self.measurement_array[imp, :, :] = self.acquire_imp_signal()
            # Check for chn overload
            no_overload = self.check_chn_overload(imp)
            # Check for force overload
            no_imp_overload = self.check_imp_overload(imp)
            # check for double imp
            imp_start, imp_end = self.get_imp_start_end(imp)
            double_ind, no_double = self.check_double_impact(imp, imp_end)
            # Check for channel overloads, double impacts
            msg = None
            if (not no_overload) and (not no_double) and (not no_imp_overload):
                msg = 'Double impact and chn #cifra# overload and force overload'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            elif (not no_overload) and (not no_double):
                msg = 'Chn #cifra# overload and double impact!'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            elif (not no_double) and (not no_imp_overload):
                msg = 'Double impact and force overload!'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            elif (not no_overload) and (not no_imp_overload):
                msg = 'Chn #cifra# overload and force overload!'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            elif not no_overload:
                msg = 'Chn #cifra# overload!'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            elif not no_double:
                msg = 'Double impact.'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            elif not no_imp_overload:
                msg = 'Force overload.'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            else:
                winsound.Beep(300, 70)
            # Plotting acquired signals
            self.plot_imp_meas(imp, msg=msg, imp_start=imp_start, imp_end=imp_end, double_ind=double_ind)
            if msg is None:
                imp += 1
                pbar.update(1)
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
        pbar.container.children[-2].style.bar_color = 'green'
        self.save_imp_test_results(save_to, pbar, series=series)

    def start_imp_test_series(self, list_of_points, measurement_file):
        self.reset_series_params()
        self.points_to_measure = list_of_points
        self.meas_file = measurement_file
        print(f'Measurement point {self.points_to_measure[self.point_ind]}')
        self.start_impact_test(save_to=self.meas_file + fr'\{self.points_to_measure[self.point_ind]}', series=True)

    def start_admittance_measurement(self, channels, impacts, chn_factors, imp_factors, save_to=None, force_chn_ind=0, existing_Y=None, existing_json=None):
        self.admittance = True
        self.adm_channels, self.adm_impacts = channels, impacts
        self.chn_factors, self.imp_factors = chn_factors, imp_factors
        self.force_chn_ind = force_chn_ind
        if (existing_Y is not None) or (existing_json is not None):  
            if (existing_Y is None) or (existing_json is None):
                raise ValueError('Both existing_Y and existing_json must be provided.')
            else:
                # load existing data
                self.Y = np.load(existing_Y)
                self.dof_data = self.load_from_json(existing_json)
                
                # check dof combinations to be measured
                to_do = [_ for _ in self.dof_data['progress'] if self.dof_data['progress'][_] == 0]
        else:
            # create new data
            self.dof_data = self.get_dof_dict()
            to_do = [_ for _ in self.dof_data['progress']]
            self.save_to_json(save_to + r'dof_data.json')

        impacts_to_do = list(set([_[2:] for _ in to_do]))
        imp_to_do = [_ for _ in self.adm_impacts if _ in impacts_to_do] # to ensure correct order of impacts
        self.start_imp_test_series(imp_to_do, save_to)

    def acquire_imp_signal(self):
        """
        Adopted from Impact testing v1.
        :return: measured data - individual measurement
        """
        trigger = pyTrigger(rows=int(self.sampling_rate * self.acquisition_time), channels=len(self.all_channels),
                            trigger_type=self.trigger_type,
                            trigger_channel=self.force_chn_ind,
                            trigger_level=self.trigger_level,
                            presamples=self.presamples)

        trig = True
        self.task.start()
        while True:
            data = self.measure()
            trigger.add_data(data.T)
            if trigger.finished:
                self.task.stop()
                break
            if trigger.triggered == True and trig == True:
                trig = False
        return trigger.get_data().T

    def measure(self):
        no_smpl_per_chn = int(self.sampling_rate * self.acquisition_time * 2)
        # factor *2 to obtain sufficient amount of samples. (?)
        data = np.array(self.task.read(number_of_samples_per_channel=no_smpl_per_chn, timeout=10.0))
        return data

    def check_chn_overload(self, imp):
        for i in range(self.measurement_array.shape[1]):
            # če vrednost preseže 95% do meje
            chn_min, chn_max = self.task.ai_channels[i].ai_min*0.95, self.task.ai_channels[i].ai_max*0.95
            sig_min, sig_max = min(self.measurement_array[imp, i, :]), max(self.measurement_array[imp, i, :])
            if (sig_min > chn_min) and (sig_max < chn_max):
                return True
            else:
                print(np.where((sig_min < chn_min) or (sig_max > chn_max)))
                return False

    def check_imp_overload(self, imp):
        imp_ampl = max(self.measurement_array[imp, self.force_chn_ind, :])
        imp_max = self.task.ai_channels[self.force_chn_ind].ai_max*0.95
        if imp_ampl < imp_max:
            return True
        else:
            return False

    def check_double_impact(self, imp, imp_end):
        ind_ = np.where(self.measurement_array[imp, self.force_chn_ind, imp_end:] > self.double_imp_force_lim)[0]
        if len(ind_) > 0:
            return ind_+imp_end, False
        else:
            return None, True

    def get_imp_start_end(self, imp):
        force = self.measurement_array[imp, self.force_chn_ind]
        max_force_ind = np.argmax(force)
        start_ = max_force_ind
        while force[start_] > self.imp_force_lim:
            if start_ > (len(force)-1)*-1:
                start_ -= 1
            else: break
        end_ = max_force_ind
        while force[end_] > self.imp_force_lim:
            if end_ < (len(force)-1):
                end_ += 1
            else:
                break
            
        return start_, end_

    # Saving and displaying results
    def save_imp_test_results(self, save_to, pbar, series=False):

         # Plot Y_done in admittance measurement
        if self.admittance:
            # get FRF
            chn_mask = np.array([_ for _ in range(self.measurement_array.shape[1]) if _ != self.force_chn_ind])
            impact_ = self.measurement_array[:, self.force_chn_ind]*self.imp_factors[self.point_ind]
            channels_ = self.measurement_array[:, chn_mask]*np.array(self.chn_factors)[None, :, None]
            imp_fft_ = np.fft.rfft(impact_).T
            chn_fft_ = np.fft.rfft(channels_).T
            if self.Y is None:
                self.Y = np.zeros((imp_fft_.shape[0], len(self.adm_channels), len(self.adm_impacts)), dtype=np.complex128)
            imp_name_ = self.points_to_measure[self.point_ind]
            imp_ind_ = self.adm_impacts.index(imp_name_)
            for chn_ in self.adm_channels:
                chn_ind_ = self.adm_channels.index(chn_)
                self.Y[:, chn_ind_, imp_ind_] = self.get_FRF(chn_fft_[:,chn_ind_], imp_fft_)
                # update dof_data
                dof_name_str_ = f'{chn_}{imp_name_}'
                self.dof_data['progress'][dof_name_str_] = 1
            np.save(self.meas_file + r'Y.npy', self.Y)
            
            self.save_to_json(self.meas_file + r'dof_data.json')

            self.get_done_matrix()
            self.plot_Y_done()

        options = [f'Measurement {i+1}' for i in range(self.no_impacts)]
        out = Output()

        # Select measurements
        selection = widgets.SelectMultiple(options=options, value=tuple(options), rows=len(options))
        selection.layout = Layout(width='200px')
        # Save measurement info
        save_meas_info_choice = widgets.ToggleButtons(
            options=['No', 'Yes'],
            description='Save measurement info: \n',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            style={'description_width': 'initial'}
        )
        # Save measurements
        button = widgets.Button(description='Save')

       

        # Buttons for measurement series
        if series:
            repeat_button = widgets.Button(style={'description_width': 'initial'},
                                           description=f'Repeat (point {self.points_to_measure[self.point_ind]})')
            try:
                next_button_text = f'Next (point {self.points_to_measure[self.point_ind + 1]})'
            except IndexError:
                next_button_text = ''

            next_button = widgets.Button(description=next_button_text)
        else:
            repeat_button, next_button = None, None

        if save_to is None:
            file_name = widgets.Text(description='Save to: ',
                                     placeholder='Filename',
                                     value='')
            widgets_ = [selection, save_meas_info_choice, file_name, button]
        else:
            if not series:
                widgets_ = selection, save_meas_info_choice, button
            else:
                widgets_ = selection, save_meas_info_choice, button, repeat_button, next_button
        display(self.widget_layout(widgets_, save_to, series))

        def save_btn_clicked(B, save_to_=save_to):
            chosen_meas = [int(_[-1])-1 for _ in list(selection.value)]
            if save_to_ is None:
                save_to_ = str(file_name.value)
                if len(save_to_) == 0:
                    message = 'Enter file name!'
                else:
                    message = f'Measurements {chosen_meas} saved to \"{save_to_}.npy\"'
                    np.save(f'{save_to_}.npy', self.measurement_array[chosen_meas, :, :])
                    if save_meas_info_choice.value == 'Yes':
                        message += f'\nMeasurement info saved to \"{save_to_}.json\"'
                        json_obj = json.dumps(self.meas_info, indent=4)
                        with open(f'{save_to_}.json', 'w') as meas_data_file:
                            meas_data_file.write(json_obj)
                    pbar.container.children[-2].style.bar_color = 'black'
            else:
                np.save(f'{save_to_}.npy', self.measurement_array[chosen_meas, :, :])
                message = f'Measurements {chosen_meas} saved to \"{save_to_}.npy\"'
                if save_meas_info_choice.value == 'Yes':
                    message += f'\nMeasurement info saved to \"{save_to_}.json\"'
                    json_obj = json.dumps(self.meas_info, indent=4)
                    with open(f'{save_to_}.json', 'w') as meas_data_file:
                        meas_data_file.write(json_obj)
                pbar.container.children[-2].style.bar_color = 'black'
            with out:
                clear_output()
                print(message)
            display(out)

        def repeat_button_clicked(B):
            clear_output(wait=True)
            print(f'Measurement point {self.points_to_measure[self.point_ind]}')
            self.start_impact_test(save_to=self.meas_file + fr'\{self.points_to_measure[self.point_ind]}', series=True)

        def next_button_clicked(B):
            try:
                next_point = self.points_to_measure[self.point_ind+1]
                self.points_measured.append(self.points_to_measure[self.point_ind])
                clear_output(wait=False)
                self.point_ind += 1
                print(f'Measurement point {self.points_to_measure[self.point_ind]}')
                self.start_impact_test(save_to=self.meas_file + fr'\{next_point}', series=True)
            except IndexError:
                print('All points measured!')
        button.on_click(save_btn_clicked)
        if series:
            repeat_button.on_click(repeat_button_clicked)
            next_button.on_click(next_button_clicked)

    @staticmethod
    def widget_layout(widgets_list, save_to, series):
        grid = GridspecLayout(2+series, 3+series)
        grid[:, 0] = widgets_list[0]
        grid[0, 1:] = widgets_list[1]
        if save_to is None:
            grid[1, 1] = widgets_list[2]
            grid[1, 2] = widgets_list[3]
        else:
            grid[1, 1:] = widgets_list[2]
        if series:
            grid[2, 1] = widgets_list[3]
            grid[2, 2] = widgets_list[4]
        return grid

    def clear_stored_data(self):
        self.measurement_array = np.zeros_like(self.measurement_array, dtype=np.float64)

    def close_task(self):
        self.task.close()

    def plot_imp_meas(self, meas_ind, msg, imp_start, imp_end, double_ind=None):
        """
        Plots individual measurement
        :param meas_ind:
        :param msg:
        :param imp_start: index of impact starting point
        :param imp_end: index of impact ending point
        :param double_ind: index of double impact location
        :return:
        """
        plot_min, plot_max = imp_start-15, imp_end+30
        if plot_min < 0:
            plot_min = 0
        if imp_start < 0:
            imp_start = 0
        # print(imp_start, imp_end)
        fig, ax = plt.subplots(1, 4, figsize=(15, 2.5), tight_layout=True)
        mask = np.where(np.array(self.all_channels) != 'force')[0]
        force_ = self.measurement_array[meas_ind, self.force_chn_ind, plot_min:plot_max].T
        times = np.arange(self.sampling_rate * self.acquisition_time) / self.sampling_rate
        ax[0].plot(times[plot_min:plot_max], force_)
        try:
            ax[0].vlines(imp_start / self.sampling_rate, min(force_) - 1, max(force_) + 1, color='green',
                         ls='--')
            ax[0].vlines(imp_end/self.sampling_rate, min(force_) - 1, max(force_) + 1, color='red', ls='--')
            ax[0].set_ylim(min(force_) - 1, max(force_) + 1)
        except ValueError:
            print(imp_start, imp_end, plot_min, plot_max)
            pass
        ax[0].set_xticks([imp_start/self.sampling_rate, imp_end/self.sampling_rate])
        ax[0].set_xticklabels([f'{_:.5f}' for _ in [imp_start / self.sampling_rate, imp_end / self.sampling_rate]])
        ax[0].set_title(f'Impact duration: {(imp_end-imp_start)/self.sampling_rate*1000:.3f} ms')
        ax[1].plot(times, self.measurement_array[meas_ind, self.force_chn_ind, :].T)
        ax[1].set_title(f'Impact amplitude: {max(force_):.3f} N')
        ax[2].plot(times, self.measurement_array[meas_ind, mask, :].T, lw=.5)
        # TODO: dodaj enoto!
        ax[2].set_title(f'Max. response: {np.max(self.measurement_array[meas_ind, mask, :]):.3f}')
        ax[3].semilogy(abs(np.fft.rfft(self.measurement_array[meas_ind, mask, :])).T, lw=.5)
        if msg is None:
            fig.suptitle(f'Impact {meas_ind + 1}/{self.no_impacts}', fontsize=15)
            fig.patch.set_facecolor('#cafac5')
        else:
            if double_ind[0] < plot_max:
                ax[0].vlines(double_ind[0]-plot_min, min(force_)-1, max(force_)+1, color='k', ls='--')
            ax[1].vlines((double_ind[0])/self.sampling_rate, min(force_) - 1, max(force_) + 1, color='k', ls='--')
            fig.suptitle(f'Impact {meas_ind + 1}/{self.no_impacts} ({msg})', fontsize=12)
            fig.patch.set_facecolor('#faa7a7')

        plt.show()

    

    # Admittance measurement methods
    def get_dof_dict(self):
        rows_, columns_ = np.meshgrid(self.adm_channels, self.adm_impacts)
        rows_, columns_ = rows_.flatten(), columns_.flatten()
        dof_dict = {}
        dof_dict['channels'], dof_dict['impacts'], dof_dict['chn_factors'], dof_dict['imp_dactors'], dof_dict['progress'] = self.adm_channels, self.adm_impacts, self.chn_factors, self.imp_factors, {}
        for r_, c_ in zip(rows_, columns_):
            dof_dict['progress'][f'{r_}{c_}'] = 0
        return dof_dict

    def get_done_matrix(self):
        self.Y_done = np.zeros((len(self.dof_data['channels']), len(self.dof_data['impacts'])))
        rows_, columns_ = np.meshgrid(self.dof_data['channels'], self.dof_data['impacts'])
        rows_, columns_ = rows_.flatten(), columns_.flatten()
        for r_, c_ in zip(rows_, columns_):
            self.Y_done[self.dof_data['channels'].index(r_), self.dof_data['impacts'].index(c_)] = self.dof_data['progress'][f'{r_}{c_}']

    def plot_Y_done(self):
        plt.figure()
        plt.imshow(self.Y_done, cmap='RdYlGn', vmin=0, vmax=1)
        plt.xlabel('Impacts')
        plt.ylabel('Channels')
        plt.xticks(np.arange(len(self.adm_impacts)), self.adm_impacts)
        plt.yticks(np.arange(len(self.adm_channels)), self.adm_channels)
        plt.show()

    
    @staticmethod
    def get_FRF(X, F, filter_list=None, estimator='H1', kind='admittance'):
        """
        Function calculates frequency response functions (FRF) from measurement data.
        :param X: np.array of accelerations (frequencies, repeated measurements)
        :param F: np.array of accelerations (frequencies, repeated measurements)
        :param filter_list: list of indices of measurements to be excluded from the FRF calculation
        :param estimator: FRF estimator (H1, H2)
        :param kind: FRF type (admittance/impedance)
        :return: averaged FRF
        """
        N = X.shape[1]
        # Izračun cenilk prenosne funkcije
        if estimator == 'H1':
            S_fx_avg = np.zeros_like(X[:, 0])
            S_ff_avg = np.zeros_like(F[:, 0])
        elif estimator == 'H2':
            S_xx_avg = np.zeros_like(X[:, 0])
            S_xf_avg = np.zeros_like(F[:, 0])
        else:
            S_fx_avg, S_ff_avg, S_xx_avg, S_xf_avg = None, None, None, None
            raise Exception('Invalid estimator. Enter H1 or H2.')
        for i in range(N):
            if estimator == 'H1':
                if filter_list is not None:
                    if i not in filter_list:
                        S_fx_avg += np.conj(F[:, i]) * X[:, i]
                        S_ff_avg += np.conj(F[:, i]) * F[:, i]
                else:
                    S_fx_avg += np.conj(F[:, i]) * X[:, i]
                    S_ff_avg += np.conj(F[:, i]) * F[:, i]
            elif estimator == 'H2':
                if filter_list is not None:
                    if i not in filter_list:
                        S_xx_avg += np.conj(X[:, i]) * X[:, i]
                        S_xf_avg += np.conj(X[:, i]) * F[:, i]
                else:
                    S_xx_avg += np.conj(X[:, i]) * X[:, i]
                    S_xf_avg += np.conj(X[:, i]) * F[:, i]
            else:
                print('Invalid estimator')
                return
        if estimator == 'H1':
            if kind == 'admittance':
                return S_fx_avg / S_ff_avg
            elif kind == 'impedance':
                return S_ff_avg / S_fx_avg
            else:
                print('Invalid FRF type')
                return
        elif estimator == 'H2':
            if kind == 'admittance':
                return S_xx_avg / S_xf_avg
            elif kind == 'impedance':
                return S_xf_avg / S_xx_avg
            else:
                print('Invalid FRF type')
                return

    @staticmethod
    def load_from_json(path):
        with open(path, 'r') as f:
            dict_ = json.load(f)
        return dict_

    def save_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.dof_data, f)

    def plot_op_meas(self):
        fig = plt.figure(figsize=(15, 2.2))
        ax = fig.add_axes([0.1, 0.1, .8, .8])
        ax.plot(np.linspace(0, self.acquisition_time, self.acquisition_time*self.sampling_rate),
                self.measurement_array.T)
        ax.set_xlabel('time [s]')
        ax.grid()
        fig.patch.set_facecolor('#cafac5')
        plt.show()

