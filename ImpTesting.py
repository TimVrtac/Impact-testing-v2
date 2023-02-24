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


class ImpactTesting:
    def __init__(self, task_name, sensor_xlsx, sensor_list, sampling_rate, samps_per_chn, acquisition_time, no_impacts,
                 trigger_type='up', trigger_level=10.0, presamples=100, imp_force_lim=0.01, double_imp_force_lim=1,
                 terminal_config=constants.TerminalConfiguration.PSEUDO_DIFF,
                 excitation_source=constants.ExcitationSource.INTERNAL,
                 current_excit_val=0.004, sample_mode=constants.AcquisitionType.CONTINUOUS):
        """
        # TODO: Trenutno samo za IEPE
        # TODO: Check for double impacts -> prilagodi beepe
        # TODO: Izrisi za vsako meritev
        # TODO: Pri nekaterih meritvah čudno poreže signale -> preveri!
        nidaqmx constants: https://nidaqmx-python.readthedocs.io/en/latest/constants.html

        # Sensor data parameters
        :param sensor_xlsx: path to Excel file with sensor data
        :param sensor_list: list of sensors (Serial numbers) or dict of shape {SN: [list of directions (x,y,z)]}

        # General channel configuration parameters
        :param terminal_config: terminal configuration parameter (DEFAULT, DIFF, NRSE,
                                PSEUFO_DIFF, RSE - see nidaqmx constants)
        :param excitation_source: excitation source parameter (EXTERNAL, INTERNAL, NONE - see nidaqmx constants)
        :param current_excit_val: excitation current [A] (float)

        # Sampling parameters
        :param sample_mode: sampling mode (CONTINUOUS, FINITE, HW_TIMED_SINGLE_POINT - see nidaqmx constants)
        :param sampling_rate: sampling rate [Hz] (int)
        :param samps_per_chn: Specifies the number of samples to acquire or generate for each channel in the task if
                              **sample_mode** is **FINITE_SAMPLES**. If **sample_mode** is **CONTINUOUS_SAMPLES**,
                               NI-DAQmx uses this value to determine the buffer size.
        :param acquisition_time: acquisition time [s] (int/float)

        # Trigger conficuration parameters
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
            self.task = nidaqmx.task.Task(new_task_name=task_name + '_')
            print(f"Repeated task name: task name changed to {task_name + '_'}")
        self.excitation_source = excitation_source
        self.current_excit_val = current_excit_val

        # General channel parameters
        self.terminal_config = terminal_config
        # nidaqmx constants - unit conversion
        self.unit_conv = {'mV/g': constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                          'mV/m/s**2': constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                          'g': constants.AccelUnits.G,
                          'm/s**2': constants.AccelUnits.METERS_PER_SECOND_SQUARED,
                          'mV/N': constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,
                          'N': constants.ForceUnits.NEWTONS}

        # add channels to the task
        self.add_channels()

        # sampling configuration
        self.sampling_rate = sampling_rate
        self.acqisition_time = acquisition_time
        self.task.timing.cfg_samp_clk_timing(rate=self.sampling_rate, sample_mode=sample_mode,
                                             samps_per_chan=samps_per_chn)  # set sampling for the task

        # list all channels
        self.all_channels = [str(_.name) for _ in self.task.ai_channels]
        self.force_chn_ind = int(np.where(np.array(self.all_channels) == 'force')[0])

        # trigger configuration
        self.trigger_type = trigger_type
        self.trigger_level = trigger_level
        self.presamples = presamples

        # measurement configuration
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
        self.meas_info = {'Sampling rate': self.sampling_rate,
                          'Acquisiton time': self.acqisition_time,
                          'Used devices': self.device_list,
                          'Channels': self.all_channels,
                          'Force channel index': self.force_chn_ind}

    # Task generation methods
    def add_channels(self):
        """
        sensors: list of sensors (Serial numbers) or dict of shape {SN: [list of directions (x,y,z)]}
        task: nidaqmx Task instance
        df: dataframe with sensor data
        """

        device_ind = 1  # pričakovan cDAQ5Mod1
        dev_chn_ind = 0
        sensor_ind = 0

        if type(self.sensor_list) == list:
            for i in self.sensor_list:
                temp_df_ = self.sensor_df[self.sensor_df['SN'].astype(str) == i]
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
                        # print(i, phys_chn, chn_name)
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
                        # print(i, phys_chn, chn_name)
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

    # Measurement methods
    def start_measurement(self, save_to=None):
        """
        :param save_to: name of the file in which measurements are to be saved.
        """
        # tqdm
        pbar = tqdm(total=self.no_impacts)

        # prevents error in case of interuption during last measurement
        if not self.task.is_task_done():
            self.task.stop()

        imp = 0
        self.clear_stored_data()
        while imp < self.no_impacts:
            # Znak za začetek meritve.
            winsound.Beep(410, 180)
            # Meritev
            self.measurement_array[imp, :, :] = self.acquire_signal()
            # Check for chn overload
            no_overload = self.check_chn_overload(imp)
            # check for double imp
            imp_start, imp_end = self.get_imp_start_end(imp)
            double_ind, no_double = self.check_double_impact(imp, imp_end)
            # Izris meritve
            msg = None
            if (not no_overload) and (not no_double):
                msg = 'Double impact and chn #cifra# overload'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            elif not no_overload:
                msg = 'Chn #cifra# overload!'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            elif not no_double:
                msg = 'Double impact.'
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            else:
                winsound.Beep(300, 70)
            self.plot_meas(imp, msg=msg, imp_start=imp_start, imp_end=imp_end, double_ind=double_ind)
            if msg is None:
                imp += 1
                pbar.update(1)
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
        pbar.container.children[-2].style.bar_color = 'green'
        self.save_results(save_to, pbar)

    def acquire_signal(self):
        """
        Iz prejšnje verzije.
        :return: measured data - individual measurement
        """
        trigger = pyTrigger(rows=int(self.sampling_rate * self.acqisition_time), channels=len(self.all_channels),
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
        no_smpl_per_chn = int(self.sampling_rate * self.acqisition_time * 2)
        # *2, da zagotovo zajamemo dovolj podatkov. Trigger gledamo samo prvo polovico časa.
        #self.task.start()
        data = np.array(self.task.read(number_of_samples_per_channel=no_smpl_per_chn, timeout=10.0))
        # self.task.wait_until_done(timeout=10)
        #self.task.stop()
        return data

    def check_chn_overload(self, imp):
        for i in range(self.measurement_array.shape[1]):
            # če vrednost preseže 95% do meje
            chn_min, chn_max = self.task.ai_channels[i].ai_min*0.95, self.task.ai_channels[i].ai_max*0.95
            sig_min, sig_max = min(self.measurement_array[imp, i, :]), max(self.measurement_array[imp, i, :])
            # print(sig_min, sig_max)
            if (sig_min > chn_min) and (sig_max < chn_max):
                return True
            else:
                print(np.where((sig_min < chn_min) or (sig_max > chn_max)))
                return False

    def check_double_impact(self, imp, imp_end):
        ind_ = np.where(self.measurement_array[imp, self.force_chn_ind, imp_end:] > self.double_imp_force_lim)[0]
        if len(ind_) > 0:
            return ind_+imp_end, False
        else:
            return None, True

    def get_imp_start_end(self, imp):
        max_force_ind = np.argmax(self.measurement_array[imp, self.force_chn_ind])
        force = self.measurement_array[imp, self.force_chn_ind]
        start_ = max_force_ind
        while force[start_] > self.imp_force_lim:
            start_ -= 1
        end_ = max_force_ind
        while force[end_] > self.imp_force_lim:
            end_ += 1
        return start_, end_

    # Saving and displaying results
    def save_results(self, save_to, pbar):
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
        if save_to is None:
            file_name = widgets.Text(description='Save to: ',
                                     placeholder='Filename',
                                     value='')
            widgets_ = [selection, save_meas_info_choice, file_name, button]
        else:
            widgets_ = selection, save_meas_info_choice, button
        display(self.widget_layout(widgets_, save_to))

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

        button.on_click(save_btn_clicked)

    @staticmethod
    def widget_layout(widgets_list, save_to):
        grid = GridspecLayout(2, 3)
        grid[:, 0] = widgets_list[0]
        grid[0, 1:] = widgets_list[1]
        if save_to is None:
            grid[1, 1] = widgets_list[2]
            grid[1, 2] = widgets_list[3]
        else:
            grid[1, 1:] = widgets_list[2]
        return grid

    def clear_stored_data(self):
        self.measurement_array = np.zeros_like(self.measurement_array, dtype=np.float64)

    def close_task(self):
        self.task.close()

    def plot_meas(self, meas_ind, msg, imp_start, imp_end, double_ind=None):
        """
        Plots individual measurement
        :param meas_ind:
        :param msg:
        :param imp_start: index of impact starting point
        :param imp_end: index of impact ending point
        :param double_ind: index of double impact location
        :return:
        """
        plot_min, plot_max = 85, 150
        # print(imp_start, imp_end)
        fig, ax = plt.subplots(1, 4, figsize=(15, 2.5), tight_layout=True)
        mask = np.where(np.array(self.all_channels) != 'force')[0]
        force_ = self.measurement_array[meas_ind, self.force_chn_ind, plot_min:plot_max].T
        ax[0].plot(force_)
        ax[0].vlines(imp_start-plot_min, min(force_)-1, max(force_)+1, color='green', ls='--')
        ax[0].vlines(imp_end-plot_min, min(force_)-1, max(force_)+1, color='red', ls='--')
        ax[0].set_ylim(min(force_)-1, max(force_)+1)
        ax[1].plot(self.measurement_array[meas_ind, self.force_chn_ind, :].T)
        ax[2].plot(self.measurement_array[meas_ind, mask, :].T)
        ax[3].semilogy(abs(np.fft.rfft(self.measurement_array[meas_ind, mask, :])).T)
        if msg is None:
            fig.suptitle(f'Impact {meas_ind + 1}/{self.no_impacts}', fontsize=15)
            fig.patch.set_facecolor('#cafac5')
        else:
            if double_ind[0] < plot_max:
                ax[0].vlines(double_ind[0]-plot_min, min(force_)-1, max(force_)+1, color='k', ls='--')
            ax[1].vlines(double_ind[0]-plot_min, min(force_) - 1, max(force_) + 1, color='k', ls='--')
            fig.suptitle(f'Impact {meas_ind + 1}/{self.no_impacts} ({msg})', fontsize=12)
            fig.patch.set_facecolor('#faa7a7')

        plt.show()
