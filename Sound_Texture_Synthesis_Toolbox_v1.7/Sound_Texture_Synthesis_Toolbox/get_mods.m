clear
% close all
clc

% in_ = sin(1:20000);

rand('state',sum(100*clock));
randn('state',sum(100*clock));

P = [];
P.audio_sr = 20000;
P.N_audio_channels = 30;
P.low_audio_f = 20;
P.hi_audio_f = 10000;
% synth_dur_smp = length(in_);
% P.imposition_windowing = 1; %1 --> unwindowed (circular boundary handling), 2 --> global window
P.measurement_windowing = 2;
P.win_steepness = .5; % must be between 0 and 1; smaller means steeper window edges
P.imposition_method = 1; %which_method=1; %1--> conj grad; 2--> gauss-newton
P.sub_imposition_order = 1; %1--> starts with subband with most power, works out from there; 2--> different starting subband each iteration
P.env_sr = 400;
P.N_mod_channels = 20; %These next four parameters control the modulation filterbank from which modulation power is measured
P.low_mod_f = 0.5; %Hz
P.hi_mod_f = 200; %Hz
P.use_more_mod_filters=0; % should be 1 if 2x overcomplete,
P.mod_filt_Q_value = 2;
P.use_zp = 0;% 0 means circular convolution; 1 means zeropadding (for modulation filtering)
P.low_mod_f_C12=1; %Hz - this is the lowest frequency in the octave-spaced modulation filterbank used for the C1 and C2 correlations

P.compression_option=1; % should be 0 for no compression, 1 for power compression, 2 for logarithmic compression
P.comp_exponent = 0.3;

P.desired_rms = .01; %.1 was too high; some files clipped during wavwrite; .01 prevents clipping but is a little low for laptop speakers
P.audio_sr = 20000;
P.max_orig_dur_s = 7;

P.orig_sound_filename = 'Applause_-_enthusiastic2.wav';
P.orig_sound_folder = 'Example_Textures/'; %must be a string, should have a slash at the end. If this is an empty string, Matlab will search its path for the file.

[temp, sr] = audioread([P.orig_sound_folder P.orig_sound_filename]);
if size(temp,2)==2
    temp = temp(:,1); %turn stereo files into mono
end
if sr ~= P.audio_sr
    temp = resample(temp, P.audio_sr, sr);
end
if rem(length(temp),2)==1
    temp = [temp; 0];
end

ds_factor=P.audio_sr/P.env_sr;

new_l = ceil(P.max_orig_dur_s*P.audio_sr/ds_factor/2)*ds_factor*2; %to accomodate downsampling of envelope
    if new_l > length(temp)
        new_l = floor(P.max_orig_dur_s*P.audio_sr/ds_factor/2)*ds_factor*2;
    end
in_ = temp(1:new_l);
in_ = in_/rms(in_)*P.desired_rms;

ds_factor=P.audio_sr/P.env_sr; %factor by which envelopes are downsampled
synth_dur_smp = ceil(length(in_)/ds_factor)*ds_factor; %ensures that length in samples is an integer multiple of envelope sr

[audio_filts, audio_cutoffs_Hz] = make_erb_cos_filters(synth_dur_smp, P.audio_sr, P.N_audio_channels, P.low_audio_f, P.hi_audio_f);

imp_win = set_measurement_window(synth_dur_smp, P.measurement_windowing, P);

%make modulation filters
if P.use_zp==1
    mod_filt_length = synth_dur_smp/ds_factor*2;
elseif P.use_zp==0
    mod_filt_length = synth_dur_smp/ds_factor;
end


[mod_filts,mod_cfreqs_Hz,mod_freqs] = make_constQ_cos_filters(mod_filt_length, P.env_sr, P.N_mod_channels, P.low_mod_f, P.hi_mod_f, P.mod_filt_Q_value);

env_ac_filts = [];
mod_C1_filts = [];
mod_C2_filts = [];
synth_S = measure_texture_stats_copy(in_, P, imp_win, audio_filts, mod_filts);

Bfig(-1)
imagesc(synth_S.mod_power)
1;